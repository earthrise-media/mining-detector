import concurrent.futures
from dataclasses import dataclass
import logging
import os
import platform
from typing import List, Optional, Tuple, Union

from descarteslabs.geo import DLTile
import ee
import geopandas as gpd
from google.api_core import retry
import numpy as np
import pandas as pd
import tensorflow as tf
import torch
import torch.nn.functional as F
from tqdm import tqdm

from tile_utils import CenteredTile, pad_patch, chips_from_tile

TileType = Union[DLTile, CenteredTile]

EE_PROJECT = os.environ.get('EE_PROJECT', 'earthindex')
ee.Initialize(opt_url="https://earthengine-highvolume.googleapis.com",
              project=EE_PROJECT)

BAND_IDS = {
    "S1": ["VV", "VH"],
    "S2L1C": ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8A", "B8", "B9", "B10", "B11", "B12"],
    "S2L2A": ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8A", "B8", "B9", "B11", "B12"],
    "EmbeddingsV1": [f"A{x:02d}" for x in range(64)]
}

SSL4EO_PATH = 'SSL4EO/pretrained/dino_vit_small_patch16_224.pt'
# Load via: torch.load(SSL4EO_PATH, weights_only=False)

if platform.system() == 'Darwin':
    tf.config.run_functions_eagerly(True)
    tf.data.experimental.enable_debug_mode()

@dataclass
class DataConfig:
    tile_size: int = 576
    tile_padding: int = 0
    collection: str = "S2L1C"
    clear_threshold: float = 0.6
    max_workers: int = 8

@dataclass
class InferenceConfig:
    pred_threshold: float = 0.5
    stride_ratio: int = 1  # stride is computed as chip_size // stride_ratio.
    tries: int = 2
    max_concurrent_tiles: int = 500
    embed_model_chip_size: int = 224
    geo_chip_size: Optional[int] = None # Required if using an embed_model
    
class GEE_Data_Extractor:
    def __init__(self, start_date: str, end_date: str, config: DataConfig):
        self.config = config
        self.bandIds = BAND_IDS.get(self.config.collection)
        self.composite = self._build_composite(start_date, end_date)

    def _build_composite(self, start_date: str, end_date: str):
        collection = self.config.collection
        clear_threshold = self.config.clear_threshold
        
        if collection == 'S2L1C':
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
        composite_tile = self.composite.clipToBoundsAndScale(
            geometry=tile_geom,
            width=tile.tilesize + 2,
            height=tile.tilesize + 2,
        )

        pixels = ee.data.computePixels(
            {
                "bandIds": self.bandIds,
                "expression": composite_tile,
                "fileFormat": "NUMPY_NDARRAY",
                # "grid": {"crsCode": tile.crs},  # caused issues in the past
            }
        )

        if isinstance(pixels, np.ndarray):
            if pixels.dtype.fields is not None:
                pixels = np.stack([pixels[band] for band in self.bandIds],
                                 axis=-1)
            else:
                pass
        else:
            pixels = np.array(pixels)

        return pixels.astype(np.float32, copy=False), tile

    def get_tile_data_concurrent(
        self, tiles: List[TileType]) -> Tuple[List[np.ndarray], List[TileType]]:
        """
        Download all tile data concurrently.
        Args:
            tiles: list of tile objects
        Returns:
            - patches: list of numpy arrays (one per tile)
            - tile_data: list of tile objects
        """
        data, tile_metadata = [], []

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers) as executor:
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
    
    def predict_on_tile(
        self, tile: TileType) -> Tuple[gpd.GeoDataFrame, Optional[TileType]]:
        """
        Predict on a single tile.
    
        Parameters
        ----------
        tile : TileType object with geometry and tilesize.

        Returns
        -------
        preds_gdf : GeoDataFrame
        failed_tile : TileType if prediction failed, else None.
        """
        try:
            pixels, tile_info = self.data_extractor.get_tile_data(tile)
        except Exception as e:
            self.logger.error(f"Error in get_tile_data, tile {tile.key}: {e}")
            return gpd.GeoDataFrame(), tile
    
        pixels = np.array(pad_patch(pixels, tile_info.tilesize))
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

        tile_width = tile_info.tilesize + 2 * tile_info.pad
        if tile_width % stride != 0:
            self.logger.warning(
                f"Padded tile width {tile_width}px is not evenly divisible "
                f"by stride {stride}px (chip_size={chip_size}, "
                f"stride_ratio={stride_ratio}). Inference may miss some pixels."
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
        preds_gdf['prob'] = mean_preds[idx]
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
                         predictions.to_file(outpath, index=False)

            self.logger.info(f"{len(fails)} failed tiles.")
            retry_tiles = fails
            tries_remaining -= 1

        return predictions


