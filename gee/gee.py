import concurrent.futures
from dataclasses import dataclass
import logging
import os
from pathlib import Path
import platform
from typing import List, Optional, Tuple, Union, ClassVar, Dict

from affine import Affine
from descarteslabs.geo import DLTile
import ee
import geopandas as gpd
from google.api_core import retry
import numpy as np
import pandas as pd
import rasterio
import tensorflow as tf
import torch
import torch.nn.functional as F
from tqdm import tqdm

from tile_utils import CenteredTile, ensure_tile_shape, chips_from_tile

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
    tilesize: int = 528
    pad: int = 24
    collection: str = "S2L1C"
    bands: Optional[List[str]] = None
    clear_threshold: float = 0.6
    max_workers: int = 8

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
    stride_ratio: int = 2  # stride is computed as chip_size // stride_ratio.
    tries: int = 2
    max_concurrent_tiles: int = 500
    embed_model_chip_size: int = 224
    geo_chip_size: Optional[int] = None # Required if using an embed_model
    cache_dir: Optional[str] = None # If given, will write/read image rasters
    
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
    def get_tile_data(self, tile: TileType) -> Tuple[np.ndarray, TileType]:
        """Download Sentinel-2 (or other collection) data for a tile.

        Inputs:
        - tile: a TileType object
        Outputs:
        - pixels: a numpy array with shape (H, W, bands)
        - tile:   the same tile object (for metadata)
        """
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
                pass
        else:
            pixels = np.array(pixels)

        pixels = ensure_tile_shape(pixels, out_size)
        return pixels.astype(np.float32, copy=False), tile

    def get_tile_data_concurrent(
        self, tiles: List[TileType]) -> Tuple[List[np.ndarray], List[TileType]]:
        """
        Download all tile data concurrently.
        Args:
            tiles: list of tile objects
        Returns:
            - data: list of numpy arrays (one per tile)
            - tile_data: list of tile objects
        """
        data, tile_metadata = [], []

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.config.max_workers) as executor:
            future_to_tile = {
                executor.submit(self.get_tile_data, tile): tile
                for tile in tiles
            }

            for future in concurrent.futures.as_completed(future_to_tile):
                tile = future_to_tile[future]
                try:
                    pixels, tile = future.result()
                    data.append(pixels)
                    tile_metadata.append(tile)
                except Exception as e:
                    print(f"Failed to fetch {tile.key}: {e}")

        return data, tile_metadata
        
    def save_tile(self, pixels: np.ndarray, tile: TileType, outdir: Path,
                  dtype="uint16") -> Path:
        pixels = np.moveaxis(pixels.astype(dtype, copy=False), -1, 0)
        bands, height, width = pixels.shape

        transform = tile.geotrans
        if isinstance(transform, tuple):
            transform = Affine(*transform)
            
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

        if self.embed_model is not None and self.config.geo_chip_size is None:
            raise ValueError(
                "geo_chip_size must be specified when using an embed_model."
            )

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        if self.embed_model is not None:
            self.embed_model = self.embed_model.to(self.device)
            self.embed_model.eval()

    def embed(self, chips: np.ndarray):
        """Embed chips via a foundation model."""
        model_chip_size = self.config.embed_model_chip_size
        geo_chip_size = self.config.geo_chip_size
        
        tensor = torch.from_numpy(chips).permute(0, 3, 1, 2)  # NHWC → NCHW
        if geo_chip_size != model_chip_size:
            tensor = F.interpolate(
                tensor, size=(model_chip_size, model_chip_size),
                mode='bicubic', align_corners=False)

        tensor = tensor.to(self.device)
        with torch.no_grad():
            outputs = self.embed_model(tensor)
            if isinstance(outputs, dict):
                outputs = outputs[list(outputs.keys())[0]]
        return outputs.cpu().numpy()

    def predict_on_tile(self, tile: TileType
                        ) -> tuple[gpd.GeoDataFrame, Optional[TileType]]:
        """
        Predict on a single tile using model and optional embed_model.
        If cache_dir is provided, tile data will be saved to / loaded from disk.
        """
        try:
            cache_dir = self.config.cache_dir
            if cache_dir is not None:
                tif_name = (
                    f"{self.data_extractor.config.collection}_"
                    f"{tile.key}_"
                    f"{self.data_extractor.start_date}_"
                    f"{self.data_extractor.end_date}.tif"
                )
                tif_path = Path(cache_dir) / tif_name

                if tif_path.exists():
                    pixels = self.data_extractor.load_tile(tif_path)
                    tile_info = tile
                else:
                    pixels, tile_info = self.data_extractor.get_tile_data(tile)
                    self.data_extractor.save_tile(pixels, tile_info, cache_dir)
            else:
                pixels, tile_info = self.data_extractor.get_tile_data(tile)

        except Exception as e:
            self.logger.error(f"Error in fetching tile {tile.key}: {e}")
            return gpd.GeoDataFrame(), tile
    
        pixels = np.clip(pixels / 10000.0, 0, 1)

        # Determine chip size
        if self.config.geo_chip_size is None:
            input_shape = self.model.layers[0].input_shape
            if isinstance(input_shape, list):  # ensemble of models
                input_shape = input_shape[0]
            chip_size = input_shape[1]
        else:
            chip_size = self.config.geo_chip_size
        stride = chip_size // self.config.stride_ratio
        
        tile_width = tile_info.tilesize + 2 * getattr(tile_info, 'pad', 0)
        if tile_width % stride != 0:
            self.logger.warning(
                f"Padded tile width {tile_width}px is not evenly divisible "
                f"by stride {stride}px (chip_size={chip_size}, "
                f"stride_ratio={self.config.stride_ratio}). "
                f"Inference may miss some pixels."
            )

        # Split into chips
        chips, chip_geoms = chips_from_tile(
            pixels, tile_info, chip_size, stride)
        chips = np.array(chips)
        chip_geoms.to_crs("EPSG:4326", inplace=True)

        try:
            if self.embed_model is None:
                preds = self.model.predict(chips, verbose=0)
            else:
                embeddings = self.embed(chips)
                preds = self.model.predict(embeddings, verbose=0)
        except Exception as e:
            self.logger.error(
                f"Error in model.predict for tile {tile_info.key}: {e}")
            return gpd.GeoDataFrame(), tile

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

        preds_gdf = gpd.GeoDataFrame(
            geometry=chip_geoms.loc[idx, "geometry"], crs="EPSG:4326")
        preds_gdf['confidence'] = mean_preds[idx]
        if preds.shape[1] > 2:
             preds_gdf["preds"] = [str(list(v)) for v in preds[idx]]

        return preds_gdf, None

    def make_predictions(self, tiles: List[TileType],
                         outpath: Optional[str] = None) -> gpd.GeoDataFrame:
        """
        Predict on the data for the tiles, with retry logic.
        Args:
            - tiles: list of tile objects
            - outpath: optional path to write predictions
        Returns:
            - predictions: GeoDataFrame of predictions
        """
        predictions = gpd.GeoDataFrame()
        retry_tiles = tiles.copy()
        tries_remaining = self.config.tries
        max_concurent_tiles = self.config.max_concurrent_tiles
        max_workers = self.data_extractor.config.max_workers

        while tries_remaining and retry_tiles:
            self.logger.info(f"{tries_remaining} tries remaining.")
            fails = []

            for i in tqdm(
                range(0, len(retry_tiles), max_concurent_tiles)):
                batch_tiles = retry_tiles[i : i + max_concurent_tiles]

                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=max_workers) as executor:
                    futures = [
                        executor.submit(self.predict_on_tile, tile)
                        for tile in batch_tiles
                    ]

                batch_predictions = []
                for future in concurrent.futures.as_completed(futures):
                    try:
                        pred_gdf, failed_tile = future.result()
                        if failed_tile is not None:
                            fails.append(failed_tile)
                        elif not pred_gdf.empty:
                            batch_predictions.append(pred_gdf)
                    except Exception as e:
                        self.logger.error(f"Tile raised exception: {e}")

                if batch_predictions:
                     batch_gdf = pd.concat(
                         batch_predictions, ignore_index=True)
                     predictions = pd.concat(
                         [predictions, batch_gdf], ignore_index=True)
                     print(f"Found {len(batch_gdf)} new positives.", flush=True)
                     self.logger.info(f"Found {len(batch_gdf)} new positives.")
                     
                     if outpath is not None:
                         Path(outpath).parent.mkdir(parents=True, exist_ok=True)
                         predictions.to_file(outpath, index=False)

            self.logger.info(f"{len(fails)} failed tiles.")
            retry_tiles = fails
            tries_remaining -= 1

        return predictions


