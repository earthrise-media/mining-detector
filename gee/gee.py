import concurrent.futures

import ee
import geopandas as gpd
import numpy as np
import pandas as pd

import utils


class S2_Data_Extractor:
    """
    Pull Sentinel-2 data for a set of tiles.
    Inputs:
        - tiles: a list of DLTile objects
        - start_date: the start date of the data
        - end_date: the end date of the data
        - clear_threshold: the threshold for cloud cover
        - batch_size: the number of tiles to process in each batch (default: 500)
    Methods: Functions are run in parallel.
        - get_data: pull the data for the tiles. Returns numpy arrays of chips
        - predict: predict on the data for the tiles. Returns a gdf of predictions and geoms
        - process_tile: Function h

    """

    def __init__(self, tiles, start_date, end_date, clear_threshold, batch_size=500):
        self.tiles = tiles
        self.start_date = start_date
        self.end_date = end_date
        self.clear_threshold = clear_threshold
        self.batch_size = batch_size

        ee.Initialize(
            opt_url="https://earthengine-highvolume.googleapis.com",
            project="earth-engine-ck",
        )

        # Harmonized Sentinel-2 Level 2A collection.
        s2 = ee.ImageCollection("COPERNICUS/S2_HARMONIZED")

        # Cloud Score+ image collection. Note Cloud Score+ is produced from Sentinel-2
        # Level 1C data and can be applied to either L1C or L2A collections.
        csPlus = ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED")
        QA_BAND = "cs_cdf"

        # Make a clear median composite.
        self.composite = (
            s2.filterDate(start_date, end_date)
            .linkCollection(csPlus, [QA_BAND])
            .map(lambda img: img.updateMask(img.select(QA_BAND).gte(clear_threshold)))
            .median()
        )

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
                #'grid': {'crsCode': tile.crs} this was causing weird issues that I believe caused problems.
            }
        )

        # convert from a structured array to a numpy array
        pixels = np.array(pixels.tolist())

        return pixels, tile

    def predict_on_tile(self, tile, model, pred_threshold=0.5):
        """
        Takes in a tile of data and a model
        Outputs a gdf of predictions and geometries
        """
        pixels, tile_info = self.get_tile_data(tile)
        pixels = np.array(utils.pad_patch(pixels, tile_info.tilesize))
        # normalize the pixels. Might need to make this flexible in the future
        pixels = np.clip(pixels / 10000.0, 0, 1)

        chip_size = model.layers[0].input_shape[1]
        stride = chip_size // 2
        chips, chip_geoms = utils.chips_from_tile(pixels, tile_info, chip_size, stride)
        chips = np.array(chips)
        chip_geoms.to_crs("EPSG:4326", inplace=True)

        # make predictions on the chips
        preds = model.predict(chips, verbose=0)
        # find the indices of the chips that are above the threshold
        pred_idx = np.where(preds > pred_threshold)[0]
        if len(pred_idx) > 0:
            # select the elements from the chip_geoms dataframe where the predictions are above the threshold
            preds_gdf = gpd.GeoDataFrame(
                geometry=chip_geoms["geometry"][pred_idx], crs="EPSG:4326"
            )
            # add the predictions to the dataframe
            preds_gdf["pred"] = preds[pred_idx]
        else:
            preds_gdf = gpd.GeoDataFrame()

        return preds_gdf, tile_info

    def get_patches(self):
        chips = []
        tile_data = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for i in range(0, len(self.tiles), self.batch_size):
                batch_tiles = self.tiles[i : i + self.batch_size]

                # Process each tile in parallel
                futures = [
                    executor.submit(self.get_tile_data, tile) for tile in batch_tiles
                ]

                # Collect the results as they become available
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    pixels, tile = result
                    chips.append(pixels)
                    tile_data.append(tile)
        return chips, tile_data

    def make_predictions(self, model, pred_threshold=0.5, batch_size=500):
        """
        Predict on the data for the tiles.
        Inputs:
            - model: a keras model
            - batch_size: the number of tiles to process in each batch (default: 500)
        Outputs:
            - predictions: a gdf of predictions and geoms
        """

        predictions = gpd.GeoDataFrame()
        evaluated_boundaries = gpd.GeoDataFrame()
        completed_tasks = 0

        with concurrent.futures.ThreadPoolExecutor() as executor:
            for i in range(0, len(self.tiles), self.batch_size):
                batch_tiles = self.tiles[i : i + self.batch_size]
                futures = [
                    executor.submit(self.predict_on_tile, tile, model, pred_threshold)
                    for tile in batch_tiles
                ]

                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    pred_gdf, tile_info = result
                    predictions = pd.concat([predictions, pred_gdf], ignore_index=True)
                    evaluated_boundaries = pd.concat(
                        [
                            evaluated_boundaries,
                            gpd.GeoDataFrame(
                                geometry=[tile_info.geometry], crs="EPSG:4326"
                            ),
                        ],
                        ignore_index=True,
                    )
                    evaluated_boundaries = evaluated_boundaries.dissolve()

                    completed_tasks += 1
                    # printing stats is a vanity metric. Also, the num preds value assumes tilesize 576, model input 48, and stride 24
                    # maybe should delete, but I like to see the progress accumulate.
                    print(
                        f"Completed {completed_tasks:,}/{len(self.tiles):,} tiles. Made {completed_tasks * 265:,} predictions. Found {len(predictions):,} positives.",
                        end="\r",
                    )
                # write data to a tmp file
                if len(predictions) > 0:
                    predictions.to_file("tmp.geojson", driver="GeoJSON")
                evaluated_boundaries.to_file("tmp_boundaries.geojson", driver="GeoJSON")

        return predictions
