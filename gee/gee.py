from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import logging
import os
from pathlib import Path
import platform
import queue
import threading
from typing import List, Literal, Optional, Tuple, Union, ClassVar, Dict, Any
import warnings

from affine import Affine
from descarteslabs.geo import DLTile
import ee
import geopandas as gpd
from google.api_core import retry
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_bounds
from rasterstats import zonal_stats
import scipy.ndimage as ndi
import tensorflow as tf
import torch
import torch.nn.functional as F
from tqdm import tqdm

from tile_utils import CenteredTile, cut_chips, create_tiles, ensure_tile_shape

TileType = Union[DLTile, CenteredTile]
PathLike = Union[str, Path]

# Repository root (directory that contains ``gee/`` and ``models/``).
REPO_ROOT = Path(__file__).resolve().parent.parent

SSL4EO_PATH = str(
    (REPO_ROOT / "models/SSL4EO/pretrained/dino_vit_small_patch16_224.pt").resolve()
)

SAM2_PATH = str((REPO_ROOT / "models/sam2").resolve())
DEFAULT_MASK_DIR = str((REPO_ROOT / "data/outputs/sam2").resolve())
DEFAULT_INFERENCE_OUTPUT_BASE = str((REPO_ROOT / "data/outputs").resolve())

# SAM2 ``build_sam2`` uses Hydra ``compose(config_name=...)`` — this must be a
# config name relative to the installed ``sam2`` package (see ``sam2.build_sam``),
# not an absolute filesystem path. Passing ``/Users/.../sam2.1_hiera_s.yaml``
# makes Hydra look for a config literally named ``Users/...`` (leading ``/`` lost).
DEFAULT_SAM2_HYDRA_CONFIG = "configs/sam2.1/sam2.1_hiera_s.yaml"

EE_PROJECT = os.environ.get('EE_PROJECT', 'earthindex')
ee.Initialize(opt_url="https://earthengine-highvolume.googleapis.com",
              project=EE_PROJECT)

if platform.system() == 'Darwin':
    tf.config.run_functions_eagerly(True)
    tf.data.experimental.enable_debug_mode()

@dataclass
class DataConfig:
    tilesize: int = 576
    pad: int = 0
    collection: str = "S2L1C"
    bands: Optional[List[str]] = None
    clear_threshold: float = 0.6
    max_workers: int = 8
    image_cache_dir: Optional[PathLike] = None #If given, will write/read images

    _BAND_IDS: ClassVar[Dict[str, List[str]]] = {
        "S1": ["VV", "VH"],
        "S2L1C": ["B1", "B2", "B3", "B4", "B5", "B6", "B7",
                   "B8A", "B8", "B9", "B10", "B11", "B12"],
        "S2L1C-12band": ["B1", "B2", "B3", "B4", "B5", "B6", "B7",
                          "B8A", "B8", "B9", "B11", "B12"],
        "S2L2A": ["B1", "B2", "B3", "B4", "B5", "B6", "B7",
                  "B8A", "B8", "B9", "B11", "B12"],
        "AlphaEarth": [f"A{x:02d}" for x in range(64)],
    }

    _NDVI_BANDS: ClassVar[Dict[str, str]] = {
        "red": "B4",
        "nir": "B8A"
    }

    def __post_init__(self):
        # Automatically resolve bands if not provided
        if self.bands is None:
            if self.collection not in self._BAND_IDS:
                raise ValueError(f"No band mapping defined for collection "
                                 f"'{self.collection}'")
            self.bands = self._BAND_IDS[self.collection]

        if self.image_cache_dir:
            self.image_cache_dir = Path(self.image_cache_dir)
            self.image_cache_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def available_collections(cls) -> List[str]:
        """Return the list of supported collection IDs."""
        return list(cls._BAND_IDS.keys())
    
@dataclass
class InferenceConfig:
    # Path to Keras classifier (.h5). Loaded inside :class:`InferenceEngine`.
    model_path: PathLike = (
        "../models/48px_v0.X-SSL4EO-MLPensemble_2025-10-21.h5"
    )
    pred_threshold: float = 0.5
    embed_model_name: Optional[str] = "ssl4eo_vit_s16"
    embed_model_path: Optional[str] = SSL4EO_PATH
    # The next 3 parameters are required if using an embedding model
    embed_model_chip_size: Optional[int] = 224
    embedding_batch_size: Optional[int] = 32
    geo_chip_size: Optional[int] = 48 
    embeddings_cache_dir: Optional[PathLike] = None
    run_sam2: bool = False
    # Base directory for prediction GeoJSONs; subfolder per model version at runtime.
    inference_output_base: PathLike = DEFAULT_INFERENCE_OUTPUT_BASE
    stride_ratio: int = 2  # stride is computed as chip_size // stride_ratio.
    tries: int = 2
    max_concurrent_tiles: int = 500

    def __post_init__(self):
        self.model_path = Path(self.model_path)
        self._validate_embedding_config()
        
        if self.embeddings_cache_dir:
            self.embeddings_cache_dir = Path(self.embeddings_cache_dir)
            self.embeddings_cache_dir.mkdir(parents=True, exist_ok=True)

        if not self.embed_model_path:
            self.embed_model_path = ""

        self.inference_output_base = Path(self.inference_output_base)

    def _validate_embedding_config(self):
        if not self.embed_model_name or not self.embed_model_path:
            return
        
        required = {
            "embed_model_chip_size": self.embed_model_chip_size,
            "embedding_batch_size": self.embedding_batch_size,
            "geo_chip_size": self.geo_chip_size,
        }

        missing = [k for k,v in required.items() if v is None]
        if missing:
            raise ValueError(
                "Embedding model enabled but missing required parameters: "
                + ", ".join(missing)
            )

@dataclass
class MaskConfig:
    prior_sigma: float = 12.0   # spatial prior sigma (pixels)
    smoothing_sigma: float = 2.5  # gaussian smoothing after upsampling (pixels)

    sam2_repo_path: PathLike = SAM2_PATH
    sam2_checkpoint: Optional[PathLike] = None
    finetuned_weights: Optional[PathLike] = None
    # Hydra config name for ``build_sam2`` (e.g. configs/sam2.1/...), not a path.
    sam2_model_cfg: Optional[PathLike] = None
    mask_dir: PathLike = DEFAULT_MASK_DIR

    def __post_init__(self):
        sam2_repo = Path(self.sam2_repo_path)
        self.sam2_repo_path = str(sam2_repo)

        # Keep these as strings: SAM2 internals expect str paths.
        self.sam2_checkpoint = str(
            Path(self.sam2_checkpoint) if self.sam2_checkpoint
            else sam2_repo / "sam2.1_hiera_small.pt"
        )
        self.finetuned_weights = str(
            Path(self.finetuned_weights) if self.finetuned_weights
            else sam2_repo / "SAM_model_96_px_final.pth"
        )
        self.sam2_model_cfg = self._resolve_sam2_hydra_config(
            self.sam2_model_cfg, sam2_repo
        )

        self.mask_dir = Path(self.mask_dir)
        self.mask_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _resolve_sam2_hydra_config(
        sam2_model_cfg: Optional[PathLike], sam2_repo: Path
    ) -> str:
        """Return Hydra ``config_name`` for ``sam2.build_sam.build_sam2``."""
        if not sam2_model_cfg:
            return DEFAULT_SAM2_HYDRA_CONFIG
        raw = str(sam2_model_cfg).strip()
        path = Path(raw).expanduser()
        configs_root = (sam2_repo / "sam2" / "configs").resolve()
        if path.is_file():
            try:
                rel = path.resolve().relative_to(configs_root)
                return f"configs/{rel.as_posix()}"
            except ValueError:
                pass
        # Already a Hydra package-relative name (or user override to experiment).
        return raw

class GEE_Data_Extractor:
    def __init__(self, start_date: str, end_date: str, config: DataConfig):
        self.start_date = str(start_date)
        self.end_date = str(end_date)
        self.config = config
        self.composite = self._build_composite(start_date, end_date)

    def _build_composite(self, start_date: str, end_date: str):
        collection = self.config.collection
        clear_threshold = self.config.clear_threshold
      
        if collection in ['S2L1C', 'S2L1C-12band']:
            s2 = ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
            csPlus = ee.ImageCollection(
                "GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED")
            QA_BAND = "cs_cdf"
            composite = (
                s2.filterDate(start_date, end_date)
                .linkCollection(csPlus, [QA_BAND])
                .map(lambda img:
                    img.updateMask(img.select(QA_BAND).gte(clear_threshold)))
                .median())

        elif collection == 'S2L2A':
            s2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            csPlus = ee.ImageCollection(
                "GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED")
            QA_BAND = "cs_cdf"
            composite = (
                s2.filterDate(start_date, end_date)
                .linkCollection(csPlus, [QA_BAND])
                .map(lambda img:
                    img.updateMask(img.select(QA_BAND).gte(clear_threshold)))
                .median())

        elif collection == 'S1':
            s1 = ee.ImageCollection("COPERNICUS/S1_GRD")
            composite = (
                s1.filterDate(start_date, end_date)
                .filter(ee.Filter.eq('instrumentMode', 'IW'))
                .filter(ee.Filter.listContains(
                    "transmitterReceiverPolarisation", "VV"))
                .filter(ee.Filter.listContains(
                    "transmitterReceiverPolarisation", "VH"))
                .mosaic())

        elif collection == 'AlphaEarth':
            emb = ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
            composite = emb.filterDate(start_date, end_date).mosaic()

        else:
            raise ValueError(f'Collection {collection} not recognized.')
        return composite

    @retry.Retry(timeout=240)
    def get_tile_data(self, tile: TileType) -> np.ndarray:
        """Download or load cached image data for a tile.

        Inputs:
        - tile: a TileType object
        Outputs:
        - pixels: a numpy array with shape (H, W, bands)
        """
        image_cache_dir = getattr(self.config, "image_cache_dir", None)
        collection = self.config.collection
        start, end = self.start_date, self.end_date
        tif_name = f"{collection}_{tile.key}_{start}_{end}.tif"

        if image_cache_dir:
            tif_path = image_cache_dir / tif_name
            if tif_path.exists():
                try:
                    return self.load_tile(tif_path)
                except Exception as e:
                    self.logger.warning(
                        f"Failed to load cached tile {tile.key}: {e}")

        tile_geom = ee.Geometry.Rectangle(tile.geometry.bounds)
        out_size = tile.tilesize + 2 * tile.pad
        composite_tile = self.composite.clipToBoundsAndScale(
            geometry=tile_geom,
            width=out_size,
            height=out_size
        )

        pixels = ee.data.computePixels(
            {
                "bandIds": self.config.bands,
                "expression": composite_tile,
                "fileFormat": "NUMPY_NDARRAY",
            }
        )

        if isinstance(pixels, np.ndarray):
            if pixels.dtype.fields is not None:
                pixels = np.stack([pixels[band] for band in self.config.bands],
                                 axis=-1)
        else:
            pixels = np.array(pixels)

        pixels = ensure_tile_shape(pixels, out_size)
        pixels = pixels.astype(np.float32, copy=False)

        if image_cache_dir:
            try:
                self.save_tile(pixels, tile, image_cache_dir)
            except Exception as e:
                self.logger.warning(f"Failed to cache tile {tile.key}: {e}")

        return pixels

    def get_tile_data_concurrent(
        self, tiles: List[TileType]) -> List[np.ndarray]:
        """
        Download all tile data concurrently.
        Args:
            tiles: list of tile objects
        Returns:
            - data: list of numpy arrays (one per tile)
        """
        data = []

        with ThreadPoolExecutor(
            max_workers=self.config.max_workers) as executor:
            data = list(executor.map(self.get_tile_data, tiles))

        return data

    @staticmethod
    def affine_from_tile(tile, width: int, height: int):
        """Construct an EPSG:4326 affine transform from tile metadata."""
        return from_bounds(*tile.geometry.bounds, width, height)

    def save_tile(
        self, pixels: np.ndarray, tile: TileType, outdir: Path,
        product_type: Literal["image", "mask", "logits"] = "image") -> Path:
        """
        Write a tile GeoTIFF in EPSG:4326.

        product_type:
            - "image"  -> satellite imagery
            - "mask"   -> uint8 segmentation mask 
            - "logits" -> float32 model logits 
        """
        if product_type == "mask":
            dtype = "uint8"
            nodata = 2
            suffix = "-msk"
        elif product_type == "logits":
            dtype = "float32"
            nodata = np.nan
            suffix = "-logits"
        else:  
            if self.config.collection[:2] == "S2":
                dtype = "uint16"
            else:
                dtype = "float32"
            nodata = None
            suffix = ""

        pixels = np.moveaxis(pixels.astype(dtype, copy=False), -1, 0)
        bands, height, width = pixels.shape

        transform = self.affine_from_tile(tile, width, height)
        crs = "EPSG:4326"

        profile = {
            "driver": "GTiff",
            "height": height,
            "width": width,
            "count": bands,
            "dtype": dtype,
            "crs": crs,
            "transform": transform,
            "compress": "deflate",
            "tiled": True,
        }

        if nodata is not None:
            profile["nodata"] = nodata

        base = (
            f"{self.config.collection}_{tile.key}_"
            f"{self.start_date}_{self.end_date}"
        )

        tif_name = f"{base}{suffix}.tif"
        outpath = outdir / tif_name

        with rasterio.open(outpath, "w", **profile) as dst:
            dst.write(pixels)

        return outpath


    def load_tile(self, path: Path) -> np.ndarray:
        """Load a tile’s pixels from GeoTIFF."""
        with rasterio.open(path) as src:
            arr = src.read().astype(np.float32)  # (B,H,W)
            arr = np.moveaxis(arr, 0, -1)  # back to (H,W,B)
        return arr

class InferenceEngine:
    """Handles tile inference, retries, and aggregation."""

    def __init__(
        self,
        start_date: str,
        end_date: str,
        data_config: DataConfig,
        config: InferenceConfig,
        mask_config: Optional[MaskConfig] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.data_extractor = GEE_Data_Extractor(
            start_date, end_date, data_config)
        self.config = config
        self.logger = logger or logging.getLogger()

        if not config.model_path:
            raise ValueError("InferenceConfig.model_path must be set to a Keras .h5 file")
        self.model = tf.keras.models.load_model(
            str(config.model_path), compile=False)

        # Add a lock to serialize model access (in-process)
        self._tf_model_lock = threading.Lock()
        
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        if config.embed_model_path:
            self.embed_model = self._load_embed_model()
        else:
            self.embed_model = None

        if self.config.run_sam2:
            if not mask_config:
                mask_config = MaskConfig()
            self.masker = SAM2_Masker(self.data_extractor, mask_config)
        else:
            self.masker = None

    def predictions_geojson_path(self, region_name: str) -> Path:
        """Path for bulk-inference positive-chip GeoJSON under ``inference_output_base``.

        ``region_name`` is a label for the output file only (e.g. ``Path(geojson).stem``);
        it is not used to load geometry.

        Uses ``config.model_path`` stem (first two ``_``-separated tokens as model
        version), ``config.pred_threshold``, ``config.inference_output_base``, and
        the data extractor's date range.
        """
        if not self.config.model_path:
            raise ValueError(
                "InferenceConfig.model_path must be set to build output path"
            )
        mp = Path(self.config.model_path)
        model_version = "_".join(mp.stem.split("_")[:2])
        period = f"{self.data_extractor.start_date}_{self.data_extractor.end_date}"
        outdir = self.config.inference_output_base / model_version
        outdir.mkdir(parents=True, exist_ok=True)
        pred = self.config.pred_threshold
        return outdir / (
            f"{region_name}_{model_version}_{pred:.2f}_{period}.geojson"
        )
            
    def _load_embed_model(self):
        if self.config.embed_model_name == "ssl4eo_vit_s16":
            embed_model = torch.load(self.config.embed_model_path,
                                     weights_only=False)
        else:
            raise ValueError(
                f"Unknown embedding model: {self.config.embed_model_name}")

        embed_model = embed_model.to(self.device)
        embed_model.eval()
        return embed_model
        
    def _make_embedding_cache_path(self, tile: TileType) -> Optional[Path]:
        """Return Path to an embedding cache file; None if disabled."""
        emb_cache_dir = getattr(self.config, "embeddings_cache_dir", None)
        if emb_cache_dir is None:
            return None

        collection = self.data_extractor.config.collection
        start = self.data_extractor.start_date
        end = self.data_extractor.end_date
        emb_name = f"{collection}_{tile.key}_{start}_{end}_embeddings.parquet"
        return emb_cache_dir / emb_name
    
    def embed(
        self,
        chips: np.ndarray,
        chip_geoms: gpd.GeoDataFrame,
        tile: Optional[TileType] = None) -> np.ndarray:
        """Embed chips via a foundation model, with optional caching."""
        
        emb_path = (
            self._make_embedding_cache_path(tile) if
            tile is not None else None
        )

        if emb_path is not None and emb_path.exists():
            try:
                gdf = gpd.read_parquet(emb_path)
                embeddings = gdf.drop(
                    columns="geometry",
                    errors='ignore').to_numpy(dtype=np.float32)
                return embeddings
            except Exception as e:
                self.logger.warning(
                    f"Failed to load cached embeddings for {tile.key}: {e}")

        model_chip_size = self.config.embed_model_chip_size
        geo_chip_size = self.config.geo_chip_size

        tensor = torch.from_numpy(chips).permute(0, 3, 1, 2)  # NHWC → NCHW
        if geo_chip_size != model_chip_size:
            tensor = F.interpolate(
                tensor, size=(model_chip_size, model_chip_size),
                mode='bicubic', align_corners=False)

        embeddings_list = []
        batch_size = self.config.embedding_batch_size
        with torch.no_grad():
            for i in range(0, len(tensor), batch_size):
                batch = tensor[i:i+batch_size].to(
                    self.device, dtype=torch.float32)
                out = self.embed_model(batch)
                if isinstance(out, dict):
                    out = out[list(out.keys())[0]]
                embeddings_list.append(out.cpu())

        embeddings = torch.cat(embeddings_list, dim=0).numpy()
                            
        if emb_path is not None:
            try:
                gdf = gpd.GeoDataFrame(
                    embeddings,
                    columns=[f"f{i}" for i in range(embeddings.shape[1])],
                    geometry=chip_geoms["geometry"],
                    crs="EPSG:4326" 
                )
                gdf.to_parquet(emb_path, index=False)
            except Exception as e:
                self.logger.warning(
                    f"Failed to save embeddings cache for {tile.key}: {e}")

        return embeddings

    def _resolve_chip_params(self):
        """Return (chip_size, stride) based on model/config."""
        if self.config.geo_chip_size is None:
            input_shape = self.model.layers[0].input_shape
            if isinstance(input_shape, list):  # ensemble of models
                input_shape = input_shape[0]
            chip_size = input_shape[1]
        else:
            chip_size = self.config.geo_chip_size
        stride = chip_size // self.config.stride_ratio
        return chip_size, stride

    def _model_infer(self, x):
        """Thread-safe wrapper around model.predict for inference."""
        with self._tf_model_lock:
            return self.model.predict(x, verbose=0)

    def _preds_to_gdf(self, preds: np.ndarray,
                      chip_geoms: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Convert model preds -> preds_gdf"""
        if preds.ndim == 2:
            if preds.shape[1] == 1:
                # sigmoid binary classifier
                mean_preds = preds.squeeze()
            elif preds.shape[1] == 2:
                # softmax binary classifier
                mean_preds = preds[:, 1]
            else:
                # ensemble of M>2 sigmoid models 
                mean_preds = np.mean(preds, axis=1)
                # (Note: M=2 ensemble would be misconstrued as softmax binary)
        else:
            # already shape (N,)
            mean_preds = preds

        idx = np.where(mean_preds > self.config.pred_threshold)[0]
        if len(idx) == 0:
            return gpd.GeoDataFrame(
                columns=["geometry", "confidence"], crs="EPSG:4326")

        preds_gdf = gpd.GeoDataFrame(
            geometry=chip_geoms.loc[idx, "geometry"].reset_index(drop=True),
            crs="EPSG:4326"
        )
        preds_gdf["confidence"] = mean_preds[idx]
        if preds.ndim == 2 and preds.shape[1] > 2:
            preds_gdf["preds"] = [str(list(v)) for v in preds[idx]]

        preds_gdf = preds_gdf.set_crs("epsg:4326", allow_override=True)
        return preds_gdf

    def _ensure_gdf(self, df):
        """Ensure input is a GeoDataFrame with geometry column and CRS.
        
        Returns: GeoDataFrame
        Raises: ValueError: If df is not a GeoDataFrame and not empty.
        """
        if isinstance(df, gpd.GeoDataFrame):
            return df
        elif df.empty:
            return gpd.GeoDataFrame(df, geometry="geometry", crs='epsg:4326')
        else:
            raise ValueError(f"Expected a gdf, got {type(df)} length {len(df)}")

    def produce_tile_input(self, tile: TileType) -> Dict[str, Any]:
        """
        Producer work for a single tile:
        - try to load cached embeddings (if embeddings_cache_dir configured)
        - if cached embeddings exist and load OK -> return {'mode':'embeddings',
            'embeddings':..., 'chip_geoms':..., 'tile': tile}
        - otherwise fetch pixels via get_tile_data -> return {'mode':'pixels',
            'pixels':..., 'tile': tile}
        Errors are raised to the caller.

        Note: When mode is ``embeddings``, the consumer runs classification only
        from cached vectors (no tile pixels in the queue). Inline SAM2 masking
        (``InferenceConfig.run_sam2``) is therefore skipped for those tiles even
        if there are positive detections. For production runs with SAM2, leave
        ``embeddings_cache_dir`` unset or avoid hitting the cache for tiles that
        need masks; use the standalone SAM2 scripts for a separate masking pass.
        """
        # Try loading cached embeddings first (best-case fast path).
        emb_path = self._make_embedding_cache_path(tile)
        if emb_path and emb_path.exists():
            try:
                embeddings_gdf = gpd.read_parquet(emb_path)
                chip_geoms = embeddings_gdf[["geometry"]].copy()
                embeddings = embeddings_gdf.drop(
                    columns="geometry",
                    errors="ignore").to_numpy(dtype=np.float32)
                return {
                    "mode": "embeddings",
                    "embeddings": embeddings,
                    "chip_geoms": chip_geoms,
                    "tile": tile
                }
            except Exception as e:
                self.logger.warning(
                    f"Failed to load embedding for {tile.key}: {e}. "
                    f"Will fetch pixels.")

        # If no usable embeddings cache, fetch pixels (I/O bound).
        pixels = self.data_extractor.get_tile_data(tile) 
        return {"mode": "pixels", "pixels": pixels, "tile": tile}

    def predict_on_tile_pixels(
        self,
        pixels: np.ndarray,
        tile: TileType) -> tuple[gpd.GeoDataFrame, Optional[TileType]]:
        """
        Run the per-tile pipeline starting from pixels (I/O already done).
        This is intended to run inside the consumer (serialized for GPU use).
        """
        chip_size, stride = self._resolve_chip_params()
        tile_width = tile.tilesize + 2 * tile.pad
        if tile_width % stride != 0:
            self.logger.warning(
                f"Padded tile width {tile_width}px is not evenly divisible "
                f"by stride {stride}px (chip_size={chip_size}, "
                f"stride_ratio={self.config.stride_ratio}). "
                f"Inference may miss some pixels."
            )

        normed = np.clip(pixels / 10000.0, 0, 1).astype(np.float32)
        chips, chip_geoms = cut_chips(
            normed, tile.geometry.bounds, chip_size, stride, crs='epsg:4326')

        try: 
            if self.embed_model is not None:
                embeddings = self.embed(chips, chip_geoms, tile)
                preds = self.predict_on_tile_embeddings(
                    embeddings, chip_geoms, tile)
                preds_gdf, failed_tile = preds
                if failed_tile is not None:
                    return gpd.GeoDataFrame(), failed_tile
            else:
                preds = self._model_infer(chips)
                preds_gdf = self._preds_to_gdf(preds, chip_geoms)

            if not preds_gdf.empty and self.masker:
                self.masker.predict(pixels, tile, preds_gdf)

            return preds_gdf, None

        except Exception as e:
            self.logger.error(f"Error predicting for tile {tile.key}: {e}")
            return gpd.GeoDataFrame(), tile

    def predict_on_tile_embeddings(
        self,
        embeddings: np.ndarray,
        chip_geoms: gpd.GeoDataFrame,
        tile: TileType) -> tuple[gpd.GeoDataFrame, Optional[TileType]]:
        """Run the TF classifier on already-available embeddings."""
        try:
            preds = self._model_infer(embeddings)
            preds_gdf = self._preds_to_gdf(preds, chip_geoms)
            return preds_gdf, None
        except Exception as e:
            tile_key = getattr(tile, "key", "unknown")
            self.logger.error(f"Error predicting on embeddings for tile "
                              f"{tile_key}: {e}")
            return gpd.GeoDataFrame(), tile

    def _consumer(
        self, q: "queue.Queue[Dict[str, Any]]", sentinel: object,
        nonlocal_predictions: List[gpd.GeoDataFrame],
        fails: List[TileType], consumer_done: threading.Event,
        lock: threading.Lock):
        """Consume items from the queue, serially running inference."""
        try:
            while True:
                item = q.get()
                if item is sentinel:
                    break

                mode = item.get("mode")
                if mode == "embeddings":
                    embeddings = item["embeddings"]
                    chip_geoms = item["chip_geoms"]
                    tile_local = item["tile"]
                    preds = self.predict_on_tile_embeddings(
                        embeddings, chip_geoms, tile_local)
                elif mode == "pixels":
                    pixels = item["pixels"]
                    tile_local = item["tile"]
                    preds = self.predict_on_tile_pixels(pixels, tile_local)
                else:
                    self.logger.error(f"Unknown queue mode: {mode}")
                    preds = (gpd.GeoDataFrame(), item.get("tile"))
                preds_gdf, failed_tile = preds
                
                with lock:
                    if failed_tile is not None:
                        fails.append(failed_tile)
                    else:
                        if not preds_gdf.empty:
                            nonlocal_predictions.append(preds_gdf)
                        
                q.task_done()

        except Exception as e:
            self.logger.error(f"Consumer error: {e}")
        finally:
            consumer_done.set()
        
    def bulk_predict(
        self,
        tiles: List[TileType],
        region_name: str,
    ) -> gpd.GeoDataFrame:
        """
        Producer-consumer bulk inference, with retry logic:
         - producers attempt to load embeddings cache; failing, fetch pixels
         - consumer serializes GPU work: embedding model (if required) and
          TF classifier

        If ``embeddings_cache_dir`` is set and a tile is served from cache
        (``mode == "embeddings"``), inline SAM2 masking does not run for that
        tile (see ``produce_tile_input``). Typical production inference without
        an embeddings cache is unaffected.

        ``region_name`` labels the output GeoJSON only (e.g. ``region_path.stem``).

        Writes cumulative predictions to the GeoJSON path from
        :meth:`predictions_geojson_path` (after each batch merge).

        """
        outpath = self.predictions_geojson_path(region_name)

        predictions = gpd.GeoDataFrame({
            "geometry": gpd.GeoSeries(dtype="geometry"),
            "confidence": gpd.pd.Series(dtype="float"),
            },
            crs="epsg:4326"
        )
        
        retry_tiles = tiles.copy()
        tries_remaining = self.config.tries
        max_concurrent_tiles = self.config.max_concurrent_tiles
        max_workers = self.data_extractor.config.max_workers

        while tries_remaining and retry_tiles:
            self.logger.info(f"{tries_remaining} tries remaining.")
            fails: List[TileType] = []

            for i in tqdm(range(0, len(retry_tiles), max_concurrent_tiles)):
                batch_tiles = retry_tiles[i : i + max_concurrent_tiles]
                q: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=64)
                sentinel = object()
                consumer_done = threading.Event()

                collected_preds: List[gpd.GeoDataFrame] = []
                lock = threading.Lock()
                
                consumer_thread = threading.Thread(
                    target=self._consumer,
                    args=(q, sentinel, collected_preds, fails,
                          consumer_done, lock),
                    daemon=True
                )
                consumer_thread.start()

                with ThreadPoolExecutor(max_workers=max_workers) as ex:
                    future_to_tile = {
                        ex.submit(self.produce_tile_input, tile):
                        tile for tile in batch_tiles
                    }
                    for fut in as_completed(future_to_tile):
                        tile = future_to_tile[fut]
                        try:
                            item = fut.result()
                            q.put(item)
                        except Exception as e:
                            self.logger.error(
                                f"Producer failed for tile {tile.key}: {e}")
                            fails.append(tile)

                q.put(sentinel)
                consumer_done.wait()

                if collected_preds:
                    batch_gdf = self._ensure_gdf(
                        gpd.pd.concat(collected_preds, ignore_index=True))
                    batch_gdf = batch_gdf.set_crs(
                        'epsg:4326', allow_override=True)
                    predictions = predictions.set_crs(
                        'epsg:4326', allow_override=True)
                    predictions = self._ensure_gdf(
                        gpd.pd.concat([predictions, batch_gdf],
                                      ignore_index=True))
                    self.logger.info(f"Found {len(batch_gdf)} new positives.")
                    print(f"Found {len(batch_gdf)} new positives.", flush=True)
                    
                    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
                    predictions.to_file(outpath, index=False)

            self.logger.info(f"{len(fails)} failed tiles.")
            retry_tiles = fails
            tries_remaining -= 1

        return predictions

    def predict_on_tile(self, tile: TileType) -> gpd.GeoDataFrame:
        """
        Convenience wrapper for debugging: run full inference on a single tile.
        Reuses the same logic as bulk_predict but without producer/consumer.
        """
        emb_cache_dir = getattr(self.config, "embeddings_cache_dir", None)
        if emb_cache_dir is not None:
            emb_path = self._make_embedding_cache_path(tile)
            if emb_path.exists():
                gdf = gpd.read_parquet(emb_path)
                chip_geoms = gdf[["geometry"]].copy()
                embeddings = gdf.drop(
                    columns="geometry",
                    errors="ignore").to_numpy(dtype=np.float32)
                preds_gdf, _ = self.predict_on_tile_embeddings(
                    embeddings, chip_geoms, tile)
                return preds_gdf

        pixels = self.data_extractor.get_tile_data(tile)
        preds_gdf, _ = self.predict_on_tile_pixels(pixels, tile)
        return preds_gdf

class SAM2_Masker:
    """Computes pixelwise segmentations."""
    def __init__(
        self, data_extractor: GEE_Data_Extractor, config: MaskConfig):
        self.data_extractor = data_extractor
        self.config = config

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self._sam_lock = threading.Lock()
        
        self.predictor = self._load_model()

    def _load_model(self):
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        
        sam2_model = build_sam2(
            self.config.sam2_model_cfg,
            self.config.sam2_checkpoint,
            device=self.device)

        state = torch.load(self.config.finetuned_weights, map_location="cpu")
        sam2_model.load_state_dict(state, strict=False)
        sam2_model.eval()
        
        return SAM2ImagePredictor(sam2_model)

    @staticmethod
    def polygon_gdf_to_pixel_bbox(
        gdf: gpd.GeoDataFrame, transform: Affine,
        width: int, height: int) -> Optional[np.ndarray]:
        """Convert georeferenced polygons to a box prompt in pixel coords."""
        if gdf.empty:
            return None

        geom = gdf.geometry.union_all()
        if geom.is_empty:
            return None

        minx, miny, maxx, maxy = geom.bounds
        inv = ~transform

        x0, y0 = inv * (minx, maxy)
        x1, y1 = inv * (maxx, miny)

        x0 = int(np.clip(np.floor(x0), 0, width - 1))
        x1 = int(np.clip(np.ceil(x1), 0, width - 1))
        y0 = int(np.clip(np.floor(y0), 0, height - 1))
        y1 = int(np.clip(np.ceil(y1), 0, height - 1))

        if x1 <= x0 or y1 <= y0:
            return None

        return np.array([x0, y0, x1, y1], dtype=np.int32)

    def get_rgb(self, pixels: np.ndarray) -> np.ndarray:
        """Convert Sentinel-2 reflectance -> uint8 RGB w/ constrast stretch."""
        rgb = pixels[..., [3, 2, 1]].astype(np.float32)
        rgb = np.clip(rgb / 3000.0, 0, 1)
        return (rgb * 255).astype(np.uint8)

    def soft_spatial_prior(
        self, preds_gdf: gpd.GeoDataFrame, transform: Affine,
        width: int, height: int) -> np.ndarray:
        """
        Piecewise spatial logit prior:
        - Inside detection mask: 0
        - Outside: quadratic negative penalty increasing with distance
        """
        raster = rasterio.features.rasterize(
            ((geom, 1) for geom in preds_gdf.geometry
            if geom is not None and not geom.is_empty),
            out_shape=(height, width),
            transform=transform,
            fill=0,
            dtype="uint8",
        ).astype(bool)

        dist_outside = ndi.distance_transform_edt(~raster)

        penalty = -(dist_outside / self.config.prior_sigma) ** 2
        prior = np.zeros_like(penalty, dtype=np.float32)
        prior[~raster] = penalty[~raster]

        return prior

    @staticmethod
    def resize_prior_to_logits(
        prior: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
        """Resample prior field to SAM2 logit resolution."""

        scale_y = target_shape[0] / prior.shape[0]
        scale_x = target_shape[1] / prior.shape[1]

        out = ndi.zoom(prior, (scale_y, scale_x), order=1)
        return out.astype(np.float32)

    def upsample_logits(
        self, logits: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
        """Resample SAM2 logits to target raster resolution, with optional
            Gaussian smoothing for spatial regularization.
        """
        logits_tensor = torch.from_numpy(logits[None, None, ...]).float()  
        upsampled = F.interpolate(
            logits_tensor,
            size=target_shape,
            mode='bilinear',
            align_corners=False
        )[0,0].numpy()

        sigma = self.config.smoothing_sigma
        if sigma and sigma > 0:
            upsampled = ndi.gaussian_filter(upsampled, sigma=sigma)

        return upsampled

    def predict(self, pixels: np.ndarray, tile: TileType,
                preds_gdf: gpd.GeoDataFrame):
        """Run SAM2 using polygon-derived box prompts.

        pixels: np.ndarray: reflectance values (H, W, B)
        tile: TileType
        preds_gdf: GeoDataFrame in EPSG:4326 (positive chip geometries).
        """
        height, width = pixels.shape[:2]
        transform = self.data_extractor.affine_from_tile(tile, width, height)

        box_prompt = self.polygon_gdf_to_pixel_bbox(
            preds_gdf, transform, width, height)

        if box_prompt is None:
            return {}

        with self._sam_lock:
            self.predictor.set_image(self.get_rgb(pixels))
            labels, scores, prob_logits = self.predictor.predict(
                box=box_prompt,
                multimask_output=True)

        best_logits = prob_logits[np.argmax(scores)]
        
        prior = self.soft_spatial_prior(preds_gdf, transform, width, height)
        prior = self.resize_prior_to_logits(prior, best_logits.shape)
        log_odds = best_logits + prior
        upsampled = self.upsample_logits(log_odds, pixels.shape[:2])
        mask = (upsampled > 0).astype('uint8')

        self.data_extractor.save_tile(
            pixels=mask[..., None],  # single band
            tile=tile,
            outdir=self.config.mask_dir,
            product_type="mask")

        self.data_extractor.save_tile(
            pixels=log_odds[..., None],  
            tile=tile,
            outdir=self.config.mask_dir,
            product_type="logits")

        return {
            "labels": labels,
            "scores": scores,
            "prob_logits": prob_logits,
            "box_prompt": box_prompt,
        }

    def _simplify_for_tiling(
        self, gdf: gpd.GeoDataFrame,
        buffer_width: float = 0.005) -> gpd.GeoDataFrame:
        """Simplify polygons for tile creation.

        Argument buffer_width defines the smoothing scale, in degrees 
            for epsg:4326 geometries.
        """
        gdf = gdf.copy()

        gdf["geometry"] = gdf.geometry.buffer(
            buffer_width, join_style=2).buffer(
            -buffer_width, join_style=2)

        def _drop_holes(geom):
            if geom.is_empty or geom is None:
                return None
            if geom.geom_type == "Polygon":
                return type(geom)(geom.exterior)
            elif geom.geom_type == "MultiPolygon":
                return type(geom)([type(p)(p.exterior) for p in geom.geoms
                                       if not p.is_empty])
            else:
                return geom

        gdf["geometry"] = gdf.geometry.map(_drop_holes)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            gdf["geometry"] = gdf.buffer(0)

        gdf = gdf[gdf.is_valid & ~gdf.is_empty].reset_index(drop=True)

        return gdf

    def bulk_mask_polygons(
        self, polys_gdf: gpd.GeoDataFrame,
        max_concurrent_tiles=500) -> gpd.GeoDataFrame:
        """Compute segmentation masks for wide-area polygons."""
        
        polys_gdf = polys_gdf.to_crs("EPSG:4326")
        region_gdf = self._simplify_for_tiling(polys_gdf)
        tiles = create_tiles(
            region_gdf.union_all(),
            self.data_extractor.config.tilesize,
            self.data_extractor.config.pad
        )
        print(f'{len(tiles)} tiles created.')

        # --- precompute intersections single-threaded - not thread safe ---
        tiles_w_polys = []
        for tile in tqdm(tiles, desc="Clipping polygons"):
            tile_polys = gpd.clip(polys_gdf, tile.geometry)
            tile_polys = tile_polys[
                tile_polys.is_valid &
                ~tile_polys.geometry.is_empty
            ]
            if not tile_polys.empty:
                tiles_w_polys.append((tile, tile_polys))

        # --- SAM2 masking multi-threaded --- 
        def process_tile(tile: TileType, tile_polys: gpd.GeoDataFrame):
            pixels = self.data_extractor.get_tile_data(tile)
            return self.predict(pixels, tile, tile_polys)

        for i in tqdm(range(0, len(tiles_w_polys), max_concurrent_tiles),
                      desc="Processing tiles"):
            batch = tiles_w_polys[i : i + max_concurrent_tiles]

            with ThreadPoolExecutor(
                max_workers=self.data_extractor.config.max_workers) as ex:
                futures = {
                    ex.submit(process_tile, tile, tp): tile for tile, tp
                        in batch
                }
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        print(f"Tile failed with error: {e}", flush=True)
    
