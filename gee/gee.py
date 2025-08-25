import concurrent.futures
import logging
import os
from typing import List, Optional, Tuple

import ee
import geopandas as gpd
import numpy as np
import pandas as pd
from google.api_core import retry
from tqdm import tqdm
import tensorflow as tf

import utils

EE_PROJECT = os.environ.get('EE_PROJECT', 'earthindex')
ee.Initialize(opt_url="https://earthengine-highvolume.googleapis.com",
              project=EE_PROJECT)

BAND_IDS = {
    "S1": ["VV", "VH"],
    "S2L1C": ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8A", "B8", "B9", "B10", "B11", "B12"],
    "S2L2A": ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8A", "B8", "B9", "B11", "B12"],
    "EmbeddingsV1": [f"A{x:02d}" for x in range(64)]
}

class GEE_Data_Extractor:
    def __init__(self, tiles, start_date, end_date, batch_size=128,
                 clear_threshold=None, collection='S2L1C', max_workers=8):
        self.tiles = tiles
        self.batch_size = batch_size
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
    def get_tile_data(self, tile):
        """Download Sentinel-2 (or other collection) data for a tile.

        Inputs:
        - tile: a DLTile object
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
                # "grid": {"crsCode": tile.crs},  # caused issues
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


    def get_patches(self) -> Tuple[List[np.ndarray], List[object]]:
        """
        Download all tile data concurrently.
        Returns:
            - chips: list of numpy arrays (one per tile)
            - tile_data: list of tile objects
        """
        chips = []
        tile_data = []

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers) as executor:
            future_to_tile = {
                executor.submit(self.get_tile_data, tile): tile
                for tile in self.tiles
            }

            for future in concurrent.futures.as_completed(future_to_tile):
                tile = future_to_tile[future]
                try:
                    pixels, tile = future.result()
                    chips.append(pixels)
                    tile_data.append(tile)
                except Exception as e:
                    print(f"Failed to fetch {tile.tile_id()}: {e}")

        return chips, tile_data

    def make_predictions(self, model, pred_threshold, stride_ratio, tries,
                         logger=None) -> gpd.GeoDataFrame:
        """
        Predict on the data for the tiles, with retry logic.
        Inputs:
            - model: a keras model
            - pred_threshold: cutoff in [0,1] for saving model predictions
            - stride_ratio: For area inference, stride = chip_size//stride_ratio
            - tries: number of times to attempt to predict on tiles
            - logger: python logging instance
        Outputs:
            - predictions: GeoDataFrame of predictions
        """
        if logger is None:
            logger = logging.getLogger()
        predictions = []
        tiles = self.tiles.copy()

        while tries and tiles:
            logger.info(f"{tries} tries remaining.")
            fails = []

            with concurrent.futures.ThreadPoolExecutor(
                max_workers=self.max_workers) as executor:
                future_to_tile = {
                    executor.submit(self.predict_on_tile, tile, model,
                                    pred_threshold, stride_ratio, logger): tile
                    for tile in tiles
                }

            for future in tqdm(
                concurrent.futures.as_completed(future_to_tile),
                total=len(future_to_tile)):
                tile = future_to_tile[future]
                try:
                    pred_gdf, failed_tile = future.result()
                    if failed_tile is not None:
                        fails.append(failed_tile)
                    else:
                        predictions_list.append(pred_gdf)
                except Exception as e:
                    logger.error(f"Tile {tile.tile_id()} raised exception: {e}")
                    fails.append(tile)

            logger.info(f"{len(fails)} tiles failed this round.")
            tiles = fails
            tries -= 1

        if predictions_list:
            predictions = pd.concat(predictions_list, ignore_index=True)
        else:
            predictions = gpd.GeoDataFrame()

        return predictions


