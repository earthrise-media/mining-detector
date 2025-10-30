from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import logging
import os
from pathlib import Path
import platform
import queue
import threading
from typing import List, Optional, Tuple, Union, ClassVar, Dict, Any
import warnings

from affine import Affine
from descarteslabs.geo import DLTile
import ee
import geopandas as gpd
from google.api_core import retry
import numpy as np
import pandas as pd
import rasterio
from rasterstats import zonal_stats
import tensorflow as tf
import torch
import torch.nn.functional as F
from tqdm import tqdm

from tile_utils import CenteredTile, chips_from_tile, create_tiles, ensure_tile_shape

TileType = Union[DLTile, CenteredTile]

EE_PROJECT = os.environ.get('EE_PROJECT', 'earthindex')
ee.Initialize(opt_url="https://earthengine-highvolume.googleapis.com",
              project=EE_PROJECT)

SSL4EO_PATH = 'SSL4EO/pretrained/dino_vit_small_patch16_224.pt'
# Load via: torch.load(SSL4EO_PATH, weights_only=False)

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
    image_cache_dir: Optional[str] = None # If given, will write/read images

    _BAND_IDS: ClassVar[Dict[str, List[str]]] = {
        "S1": ["VV", "VH"],
        "S2L1C": ["B1", "B2", "B3", "B4", "B5", "B6", "B7",
                   "B8A", "B8", "B9", "B10", "B11", "B12"],
        "S2L1C-12band": ["B1", "B2", "B3", "B4", "B5", "B6", "B7",
                          "B8A", "B8", "B9", "B11", "B12"],
        "S2L2A": ["B1", "B2", "B3", "B4", "B5", "B6", "B7",
                  "B8A", "B8", "B9", "B11", "B12"],
        "EmbeddingsV1": [f"A{x:02d}" for x in range(64)],
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

    @classmethod
    def available_collections(cls) -> List[str]:
        """Return the list of supported collection IDs."""
        return list(cls._BAND_IDS.keys())
    
@dataclass
class InferenceConfig:
    pred_threshold: float = 0.5
    dissolve_threshold: float = 0.6
    ndvi_threshold: Optional[float] = None
    dissolve_buffer_deg: float = 0.00001
    stride_ratio: int = 2  # stride is computed as chip_size // stride_ratio.
    tries: int = 2
    max_concurrent_tiles: int = 500
    embed_model_chip_size: int = 224
    embedding_batch_size: int = 32
    geo_chip_size: Optional[int] = None # Required if using an embed_model
    embeddings_cache_dir: Optional[str] = None

    
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

        elif collection == 'EmbeddingsV1':
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
            tif_path = Path(image_cache_dir) / tif_name
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
            height=out_size,
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
    
    def _get_affine_transform(
        self,
        transform: Union[Affine,
                   Tuple[float, float, float, float, float, float]]) -> Affine:
        """
        Convert DLTile-style transform or Affine object to rasterio Affine.
        """
        if isinstance(transform, Affine):
            return transform
        elif isinstance(transform, tuple) and len(transform) == 6:
            x_min, x_res, x_rot, y_max, y_rot, y_res = transform
            return Affine(x_res, x_rot, x_min, y_rot, y_res, y_max)
        else:
            raise TypeError(f"Unexpected transform type: {type(transform)}")
        
    def save_tile(self, pixels: np.ndarray, tile: TileType, outdir: Path,
                  dtype="uint16") -> Path:
        pixels = np.moveaxis(pixels.astype(dtype, copy=False), -1, 0)
        bands, height, width = pixels.shape

        transform = self._get_affine_transform(tile.geotrans)
        profile = {
            "driver": "GTiff",
            "height": height,
            "width": width,
            "count": bands,
            "dtype": dtype,
            "crs": tile.crs,
            "transform": transform,
            "compress": "deflate",
        }
        tif_name = (f"{self.config.collection}_{tile.key}_"
                    f"{self.start_date}_{self.end_date}.tif")
        outpath = Path(outdir) / tif_name
        outpath.parent.mkdir(parents=True, exist_ok=True)
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
        data_extractor: GEE_Data_Extractor,
        model: tf.keras.Model,
        config: InferenceConfig,
        embed_model: Optional[torch.nn.Module] = None,
        logger: Optional[logging.Logger] = None):
        
        self.data_extractor = data_extractor
        self.model = model
        self.config = config
        self.embed_model = embed_model
        self.logger = logger or logging.getLogger()

        if config.ndvi_threshold:
            self.masker = Masker(data_extractor, config.ndvi_threshold)

        if self.embed_model is not None and self.config.geo_chip_size is None:
            raise ValueError(
                "geo_chip_size must be specified when using an embed_model."
            )

        # Add a lock to serialize model access (in-process)
        self._tf_model_lock = threading.Lock()
        
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        if self.embed_model is not None:
            self.embed_model = self.embed_model.to(self.device)
            self.embed_model.eval()

    def _make_embedding_cache_path(self, tile: TileType) -> Optional[Path]:
        """Return Path to an embedding cache file; None if disabled."""
        emb_cache_dir = getattr(self.config, "embeddings_cache_dir", None)
        if emb_cache_dir is None:
            return None

        emb_cache_dir = Path(emb_cache_dir)
        emb_cache_dir.mkdir(parents=True, exist_ok=True)

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

    def _dissolve(self, preds_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Dissolve overlapping predicted chips into polygons."""
        if preds_gdf.empty:
            return preds_gdf

        thresh = self.config.dissolve_threshold
        df = preds_gdf[preds_gdf["confidence"] > thresh].copy()
        if df.empty:
            return gpd.GeoDataFrame(
                columns=["geometry", "confidence"], crs=preds_gdf.crs)

        buffer_width = self.config.dissolve_buffer_deg
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            dissolved = df.buffer(buffer_width, join_style=2).unary_union
            polys_gdf = gpd.GeoDataFrame(geometry=[dissolved],
                                         crs=preds_gdf.crs)
            polys_gdf = polys_gdf.explode(index_parts=False).reset_index(
                drop=True)
            polys_gdf.geometry = polys_gdf.buffer(-buffer_width, join_style=2)

        joined = gpd.sjoin(df, polys_gdf, predicate="intersects", how="inner")
        mean_conf = joined.groupby("index_right")["confidence"].mean()
        polys_gdf["confidence"] = polys_gdf.index.map(mean_conf)
        return polys_gdf

    def predict_on_tile_pixels(
        self,
        pixels: np.ndarray,
        tile: TileType) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame,
                                 Optional[TileType]]:
        """
        Run the per-tile pipeline starting from pixels (I/O already done).
        This is intended to run inside the consumer (serialized for GPU use).
        """
        pixels = np.clip(pixels / 10000.0, 0, 1)

        chip_size, stride = self._resolve_chip_params()
        tile_width = tile.tilesize + 2 * tile.pad
        if tile_width % stride != 0:
            self.logger.warning(
                f"Padded tile width {tile_width}px is not evenly divisible "
                f"by stride {stride}px (chip_size={chip_size}, "
                f"stride_ratio={self.config.stride_ratio}). "
                f"Inference may miss some pixels."
            )

        chips, chip_geoms = chips_from_tile(pixels, tile, chip_size, stride)
        chips = np.array(chips, dtype=np.float32)
        chip_geoms.to_crs("EPSG:4326", inplace=True)

        try: 
            if self.embed_model is not None:
                embeddings = self.embed(chips, chip_geoms, tile)
                preds = self.predict_on_tile_embeddings(
                    embeddings, chip_geoms, tile)
                preds_gdf, polys_gdf, failed_tile = preds
                if failed_tile is not None:
                    return gpd.GeoDataFrame(), gpd.GeoDataFrame(), failed_tile
            else:
                preds = self._model_infer(chips)
                preds_gdf = self._preds_to_gdf(preds, chip_geoms)
                polys_gdf = self._dissolve(preds_gdf)

            if self.config.ndvi_threshold is not None and not polys_gdf.empty:
                ndvi = self.masker.compute_ndvi(pixels)
                mask = (ndvi < self.config.ndvi_threshold).astype(np.uint8)
                polys_gdf = self.masker.compute_masked_area(
                    polys_gdf, mask, tile)

            return preds_gdf, polys_gdf, None

        except Exception as e:
            self.logger.error(f"Error predicting for tile {tile.key}: {e}")
            return gpd.GeoDataFrame(), gpd.GeoDataFrame(), tile

    def predict_on_tile_embeddings(
        self,
        embeddings: np.ndarray,
        chip_geoms: gpd.GeoDataFrame,
        tile: Optional[TileType]) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame,
                                           Optional[TileType]]:
        """Run the TF classifier on already-available embeddings."""
        try:
            preds = self._model_infer(embeddings)
            preds_gdf = self._preds_to_gdf(preds, chip_geoms)
            polys_gdf = self._dissolve(preds_gdf)
            return preds_gdf, polys_gdf, None
        except Exception as e:
            tile_key = getattr(tile, "key", "unknown")
            self.logger.error(f"Error predicting on embeddings for tile "
                              f"{tile_key}: {e}")
            return gpd.GeoDataFrame(), gpd.GeoDataFrame(), tile

    def _consumer(
        self, q: "queue.Queue[Dict[str, Any]]", sentinel: object,
        nonlocal_predictions: List[gpd.GeoDataFrame],
        nonlocal_polys: List[gpd.GeoDataFrame],
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
                    preds = (gpd.GeoDataFrame(), gpd.GeoDataFrame(),
                             item.get("tile"))
                preds_gdf, polys_gdf, failed_tile = preds
                
                with lock:
                    if failed_tile is not None:
                        fails.append(failed_tile)
                    else:
                        if not preds_gdf.empty:
                            nonlocal_predictions.append(preds_gdf)
                        if not polys_gdf.empty:
                            nonlocal_polys.append(polys_gdf)
                        
                q.task_done()

        except Exception as e:
            self.logger.error(f"Consumer error: {e}")
        finally:
            consumer_done.set()
        
    def bulk_predict(
        self, tiles: List[TileType],
        outpath: Optional[str] = None) -> tuple[gpd.GeoDataFrame,
                                                gpd.GeoDataFrame]:
        """
        Producer-consumer bulk inference, with retry logic:
         - producers attempt to load embeddings cache; failing, fetch pixels
         - consumer serializes GPU work: embedding model (if required) and 
          TF classifier

        """
        predictions = gpd.GeoDataFrame({
            "geometry": gpd.GeoSeries(dtype="geometry"),
            "confidence": gpd.pd.Series(dtype="float"),
            },
            crs="epsg:4326"
        )
        polys = gpd.GeoDataFrame({
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
                collected_polys: List[gpd.GeoDataFrame] = []
                lock = threading.Lock()
                
                consumer_thread = threading.Thread(
                    target=self._consumer,
                    args=(q, sentinel, collected_preds, collected_polys,
                          fails, consumer_done, lock),
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
                    
                    if outpath is not None:
                        Path(outpath).parent.mkdir(parents=True, exist_ok=True)
                        predictions.to_file(outpath, index=False)
                        
                if collected_polys:
                    batch_polys_gdf = self._ensure_gdf(
                        gpd.pd.concat(collected_polys, ignore_index=True))
                    batch_polys_gdf = batch_polys_gdf.set_crs(
                        'epsg:4326', allow_override=True)
                    polys = polys.set_crs(
                        'epsg:4326', allow_override=True)
                    polys = self._ensure_gdf(
                        gpd.pd.concat([polys, batch_polys_gdf],
                                      ignore_index=True))

                    if outpath is not None:
                        dissolved_path = Path(outpath).with_name(
                            f"{Path(outpath).stem}-"
                            f"dissolved{self.config.dissolve_threshold}.geojson"
                        )
                        polys.to_file(dissolved_path, index=False)

            self.logger.info(f"{len(fails)} failed tiles.")
            retry_tiles = fails
            tries_remaining -= 1

        return predictions, polys

    def predict_on_tile(
        self, tile: TileType) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
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
                preds_gdf, polys_gdf, _ = self.predict_on_tile_embeddings(
                    embeddings, chip_geoms, tile)
                return preds_gdf, polys_gdf

        pixels = self.data_extractor.get_tile_data(tile)
        preds_gdf, polys_gdf, _ = self.predict_on_tile_pixels(pixels, tile)
        return preds_gdf, polys_gdf

class Masker:
    """Computes pixel-based masks and masked areas for polygons."""
    def __init__(
        self, data_extractor: GEE_Data_Extractor, ndvi_threshold: float):
        self.data_extractor = data_extractor
        self.ndvi_threshold = ndvi_threshold

    def compute_ndvi(self, pixels: np.ndarray) -> np.ndarray:
        """Compute NDVI from pixel array."""
        red_idx = self.data_extractor.config.bands.index(
            self.data_extractor.config._NDVI_BANDS['red'])
        nir_idx = self.data_extractor.config.bands.index(
            self.data_extractor.config._NDVI_BANDS['nir'])
        red = pixels[:, :, red_idx]
        nir = pixels[:, :, nir_idx]
        return (nir - red) / (nir + red + 1e-6)

    def compute_masked_area(
        self,
        polys_gdf: gpd.GeoDataFrame,
        mask: np.ndarray,
        tile: TileType) -> gpd.GeoDataFrame:
        """Compute masked area for polygons given a binary mask."""
        polys_gdf = polys_gdf[~polys_gdf.geometry.is_empty].copy()
        polys_proj = polys_gdf.to_crs(tile.crs)
        polys_gdf["Polygon area (ha)"] = polys_proj.geometry.area / 10_000.0

        transform = self.data_extractor._get_affine_transform(tile.geotrans)
        stats = zonal_stats(
            polys_proj.geometry,
            mask.astype(np.uint8),
            affine=transform,
            categorical=True,
            all_touched=False
        )

        polys_gdf["Mined area (ha)"] = np.array([s.get(1, 0) for s in stats],
                                                dtype=np.float32) / 100.0
        return polys_gdf

    def dissolve(
        self,
        polys_gdf: gpd.GeoDataFrame,
        area_fields: List[str] = ["Polygon area (ha)", "Mined area (ha)"],
        conf_field: str = "confidence",
        buffer_deg: float = 0.00001) -> gpd.GeoDataFrame:
        """Dissolve polygons using buffer+sjoin and aggregate attributes."""
        gdf = polys_gdf.copy()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            dissolved_geom = gdf.buffer(buffer_deg, join_style=2).unary_union
            dissolved = gpd.GeoDataFrame(geometry=[dissolved_geom],
                                         crs=polys_gdf.crs)
            dissolved = dissolved.explode(index_parts=False).reset_index(
                drop=True)
            dissolved.geometry = dissolved.buffer(-buffer_deg, join_style=2)

        joined = gpd.sjoin(gdf, dissolved, how="inner", predicate="intersects")
        grouped = joined.groupby("index_right")

        records = []
        for idx, group in grouped:
            rec = {}
            rec["geometry"] = dissolved.loc[idx, "geometry"]
            for af in area_fields:
                rec[af] = group[af].sum()
                
            # Weighted confidence average
            if conf_field in gdf.columns:
                weights = group[area_fields[0]]
                if weights.sum() > 0:
                    rec[conf_field] = ((group[conf_field] * weights).sum() /
                        weights.sum())
                else:
                    rec[conf_field] = group[conf_field].mean()
                    
            records.append(rec)

        return gpd.GeoDataFrame(records, crs=polys_gdf.crs)

    def _simplify_for_tiling(
        self, gdf: gpd.GeoDataFrame, tol: float = 0.01) -> gpd.GeoDataFrame:
        """Simplify polygons for tile creation."""
        gdf = gdf.copy()
        gdf["geometry"] = gdf.geometry.simplify(tol, preserve_topology=True)

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

    
    def ndvi_mask_polygons(
        self, polys_gdf: gpd.GeoDataFrame,
        max_concurrent_tiles=500) -> gpd.GeoDataFrame:
        """Compute NDVI-based masked area for polygons."""
        polys_gdf = polys_gdf.to_crs("EPSG:4326")
        region = polys_gdf.unary_union

        region_gdf = self._simplify_for_tiling(polys_gdf)
        tiles = create_tiles(
            region_gdf.unary_union,
            self.data_extractor.config.tilesize,
            self.data_extractor.config.pad
        )
        print(f'{len(tiles)} tiles created.')

        def process_tile(tile: TileType):
            tile_polys = gpd.clip(polys_gdf, tile.geometry)
            if tile_polys.empty:
                return gpd.GeoDataFrame(
                    columns=polys_gdf.columns, crs=polys_gdf.crs)
            pixels = self.data_extractor.get_tile_data(tile)
            ndvi = self.compute_ndvi(pixels)
            mask = (ndvi < self.ndvi_threshold).astype(np.uint8)
            return self.compute_masked_area(tile_polys, mask, tile)

        results = []
        for i in tqdm(range(0, len(tiles), max_concurrent_tiles),
                      desc="Processing tiles"):
            batch_tiles = tiles[i : i + max_concurrent_tiles]
            batch_results = []

            """
            with ThreadPoolExecutor(
                max_workers=self.data_extractor.config.max_workers) as ex:
                futures = {
                    ex.submit(process_tile, tile): tile for tile in batch_tiles
                }
                for future in as_completed(futures):
                    try:
                        masked = future.result()
                        if masked is not None and not masked.empty:
                            batch_results.append(masked)
                    except Exception as e:
                        print(f"Tile failed with error: {e}", flush=True)
            """
            for tile in tqdm(batch_tiles):
                masked = process_tile(tile)
                if masked is not None and not masked.empty:
                    batch_results.append(masked)

            if batch_results:
                batch_gdf = pd.concat(batch_results, ignore_index=True)
                results.append(batch_gdf)

        if results:
            masked_polys = gpd.GeoDataFrame(
                pd.concat(results, ignore_index=True), crs=polys_gdf.crs)
            masked_polys = self.dissolve(masked_polys)
            return masked_polys
        else:
            return gpd.GeoDataFrame(
                columns=polys_gdf.columns, crs=polys_gdf.crs)
