import concurrent.futures
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
from tensorflow.keras import Model
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

if platform.system() == 'Darwin':
    tf.config.run_functions_eagerly(True)
    tf.data.experimental.enable_debug_mode()

class GEE_Data_Extractor:
    def __init__(self, start_date, end_date, clear_threshold=None,
                 collection='S2L1C', max_workers=8):
        self.bandIds = BAND_IDS.get(collection)
        self.collection = collection
        self.max_workers = max_workers 
        self.composite = self._build_composite(
            collection, start_date, end_date, clear_threshold)

    def _build_composite(self, collection, start_date, end_date,
                         clear_threshold):
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

    def predict_on_tile(
        self,
        tile: TileType,
        model: Model,
        pred_threshold: float,
        stride_ratio: int,
        logger: logging.Logger,
    ) -> Tuple[gpd.GeoDataFrame, Optional[TileType]]:
        """
        Predict on a single tile.
    
        Parameters
        ----------
        tile : TileType object with geometry and tilesize.
        model : a keras Model
        pred_threshold : float Cutoff for considering a prediction positive.
        stride_ratio : int 
            Stride ratio for sliding window: stride = chip_size // stride_ratio.
        logger : logging.Logger instance.

        Returns
        -------
        preds_gdf : GeoDataFrame
        failed_tile : TileType if prediction failed, else None.
        """
        try:
            pixels, tile_info = self.get_tile_data(tile)
        except Exception as e:
            logger.error(f"Error in get_tile_data for tile {tile.key}: {e}")
            return gpd.GeoDataFrame(), tile
    
        pixels = np.array(pad_patch(pixels, tile_info.tilesize))
        pixels = np.clip(pixels / 10000.0, 0, 1)

        # Determine chip size
        input_shape = model.layers[0].input_shape
        if isinstance(input_shape, list):  # ensemble of models
            input_shape = input_shape[0]
        chip_size = input_shape[1]
        stride = chip_size // stride_ratio

        # Split into chips
        chips, chip_geoms = chips_from_tile(
            pixels, tile_info, chip_size, stride)
        chips = np.array(chips)
        chip_geoms.to_crs("EPSG:4326", inplace=True)

        try:
            preds = model.predict(chips, verbose=0)
        except Exception as e:
            logger.error(
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

        idx = np.where(mean_preds > pred_threshold)[0]

        preds_gdf = gpd.GeoDataFrame(
            geometry=chip_geoms.loc[idx, "geometry"], crs="EPSG:4326")
        preds_gdf['mean'] = mean_preds[idx]
        if preds.shape[1] > 2:
             preds_gdf["preds"] = [str(list(v)) for v in preds[idx]]

        return preds_gdf, None

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

    def make_predictions(
        self,
        tiles: List[TileType],
        model: Model,
        pred_threshold: float,
        stride_ratio: int = 1,
        tries: int = 2,
        batch_size: int = 500,
        logger: logging.Logger = None,
    ) -> gpd.GeoDataFrame:
        """
        Predict on the data for the tiles, with retry logic.
        Args:
            - tiles: list of tile objects
            - model: a keras model
            - pred_threshold: cutoff in [0,1] for saving model predictions
            - stride_ratio: For area inference, stride = chip_size//stride_ratio
            - tries: number of times to attempt to predict on tiles
            - batch_size: limit on concurrent futures for inference
            - logger: python logging instance
        Returns:
            - predictions: GeoDataFrame of predictions
        """
        if logger is None:
            logger = logging.getLogger()
        predictions = gpd.GeoDataFrame()
        retry_tiles = tiles.copy()

        while tries and retry_tiles:
            logger.info(f"{tries} tries remaining.")
            fails = []

            for i in tqdm(range(0, len(retry_tiles), batch_size)):
                batch_tiles = retry_tiles[i : i + batch_size]

                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=self.max_workers) as executor:
                    futures = [
                        executor.submit(
                            self.predict_on_tile, tile, model,
                            pred_threshold, stride_ratio, logger
                        )
                        for tile in batch_tiles
                    ]

                for future in concurrent.futures.as_completed(futures):
                    try:
                        pred_gdf, failed_tile = future.result()
                        if failed_tile is not None:
                            fails.append(failed_tile)
                        elif not pred_gdf.empty:
                            predictions = pd.concat(
                                [predictions, pred_gdf], ignore_index=True)

                    except Exception as e:
                        logger.error(f"Tile raised exception: {e}")

                print(f"Found {len(predictions)} total positives.", flush=True)

            logger.info(f"{len(fails)} failed tiles.")
            retry_tiles = fails
            tries -= 1

        return predictions


