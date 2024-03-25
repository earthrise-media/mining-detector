import concurrent.futures
import os

import ee
import geopandas as gpd
from google.api_core import retry
import tensorflow as tf
import numpy as np
import pandas as pd
from tqdm import tqdm

import utils

EE_PROJECT = os.environ.get('EE_PROJECT', 'earthindex')

class S2_Data_Extractor:
    """
    Pull Sentinel-2 data for a set of tiles.
    Inputs:
        - tiles: a list of DLTile objects
        - start_date: the start date of the data
        - end_date: the end date of the data
        - clear_threshold: the threshold for cloud cover
        - batch_size: the number of tiles to process in each batch
    Methods: Functions are run in parallel.
        - get_data: pull the data for the tiles. Returns numpy arrays of chips
        - predict: predict on the data for the tiles. Returns a gdf of predictions and geoms
        - process_tile: Function h

    """

    def __init__(self, tiles, start_date, end_date, clear_threshold,
                 batch_size, ee_project=EE_PROJECT):
        self.tiles = tiles
        self.start_date = start_date
        self.end_date = end_date
        self.clear_threshold = clear_threshold
        self.batch_size = batch_size

        ee.Initialize(
            opt_url="https://earthengine-highvolume.googleapis.com",
            project=ee_project,
        )

        s2 = ee.ImageCollection("COPERNICUS/S2_HARMONIZED")

        # Cloud Score+ is produced from L1C data;  can be applied to L1C or L2A.
        csPlus = ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED")
        QA_BAND = "cs_cdf"

        self.composite = (
            s2.filterDate(start_date, end_date)
            .linkCollection(csPlus, [QA_BAND])
            .map(lambda img:
                     img.updateMask(img.select(QA_BAND).gte(clear_threshold)))
            .median()
        )
    
    @retry.Retry(timeout=240)
    def get_tile_data(self, tile):
        """
        Download Sentinel-2 data for a tile.
        Inputs:
            - tile: a DLTile object
            - composite: a Sentinel-2 image collection
        Outputs:
            - pixels: a numpy array containing the Sentinel-2 data
        """
        tile_geom = ee.Geometry.Rectangle(tile.geometry.bounds)
        composite_tile = self.composite.clipToBoundsAndScale(
            geometry=tile_geom, width=tile.tilesize + 2, height=tile.tilesize + 2
        )
        pixels = ee.data.computePixels(
            {
                "bandIds": [
                    "B1",
                    "B2",
                    "B3",
                    "B4",
                    "B5",
                    "B6",
                    "B7",
                    "B8A",
                    "B8",
                    "B9",
                    "B11",
                    "B12",
                ],
                "expression": composite_tile,
                "fileFormat": "NUMPY_NDARRAY",
                #'grid': {'crsCode': tile.crs} this was causing weird issues
            }
        )

        # convert from a structured array to a numpy array
        pixels = np.array(pixels.tolist())

        return pixels, tile
    
    def predict_on_tile(self, tile, model, pred_threshold, logger):
        """
        Takes in a tile of data and a model
        Outputs a gdf of predictions, and with an exception, the tile
        """
        try:
            pixels, tile_info = self.get_tile_data(tile)
        except Exception as e:
            logger.error(f"Error in get_tile_data for tile {tile.key}: {e}")
            return gpd.GeoDataFrame(), tile
        
        pixels = np.array(utils.pad_patch(pixels, tile_info.tilesize))
        pixels = np.clip(pixels / 10000.0, 0, 1)

        input_shape = model.layers[0].input_shape
        if type(input_shape) == list:  # For an ensemble of models
            input_shape = input_shape[0]
        chip_size = input_shape[1]
        
        stride = chip_size // 2
        chips, chip_geoms = utils.chips_from_tile(
            pixels, tile_info, chip_size, stride)
        chips = np.array(chips)
        chip_geoms.to_crs("EPSG:4326", inplace=True)

        try:
            preds = model.predict(chips, verbose=0)
            idx = np.where(np.mean(preds, axis=1) > pred_threshold)[0]
        except Exception as e:
            logger.error(f"Error in model.predict for tile {tile_info}: {e}")
            return gpd.GeoDataFrame(), tile

        preds_gdf = gpd.GeoDataFrame(
            geometry=chip_geoms.loc[idx, "geometry"], crs="EPSG:4326")
        preds_gdf['mean'] = np.mean(preds[idx], axis=1)
        preds_gdf['preds'] = [str(list(v)) for v in preds[idx]]
        
        return preds_gdf, None

    def get_patches(self):
        chips = []
        tile_data = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for i in range(0, len(self.tiles), self.batch_size):
                batch_tiles = self.tiles[i : i + self.batch_size]

                futures = [executor.submit(self.get_tile_data, tile)
                               for tile in batch_tiles]

                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    pixels, tile = result
                    chips.append(pixels)
                    tile_data.append(tile)
        return chips, tile_data

    def make_predictions(self, model, pred_threshold, tries, logger=None):
        """
        Predict on the data for the tiles.
        Inputs:
            - model: a keras model
            - pred_threshold: cutoff in [0,1] for saving model predictions
            - tries: number of times to attempt to predict on tiles
            - logger: python logging instance
        Outputs:
            - predictions: a gdf of predictions
        """
        if not logger:
            logger = logging.getLogger()
        predictions = gpd.GeoDataFrame()
        tiles = self.tiles.copy()
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            while tries:
                logger.info(f"{tries} tries remaining.")
                fails = []
                for i in tqdm(range(0, len(tiles), self.batch_size)):
                    batch_tiles = tiles[i : i + self.batch_size]
                    futures = [
                        executor.submit(self.predict_on_tile, tile, model,
                                        pred_threshold, logger)
                        for tile in batch_tiles
                    ]

                    for future in concurrent.futures.as_completed(futures):
                        pred_gdf, failed_tile = future.result()
                        if failed_tile is not None:
                            fails.append(failed_tile)
                        else:
                            predictions = pd.concat(
                                [predictions, pred_gdf], ignore_index=True)

                    print(f"Found {len(predictions)} positives.")
                logger.info(f"{len(fails)} failed tiles.")
                tiles = fails.copy()
                tries -= 1

        return predictions

