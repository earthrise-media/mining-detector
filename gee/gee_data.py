import concurrent.futures

import ee
import geopandas as gpd

import numpy as np
import shapely
import ee
import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from descarteslabs.geo import DLTile
import gc

ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')

def create_tiles(region, tilesize, padding):
    """
    Create a set of tiles that cover a region.
    Inputs:
        - region: a geojson polygon
        - tilesize: the size of the tiles in pixels
        - padding: the number of pixels to pad each tile
    Outputs:
        - tiles: a list of DLTile objects
    """
    tiles = DLTile.iter_from_shape(region, tilesize=tilesize, resolution=10, pad=padding)
    tiles = [tile for tile in tiles]
    return tiles


def get_image_data(tiles, start_date, end_date, model, pred_threshold=0.5, clear_threshold=0.6, batch_size=500):
    """
    Download Sentinel-2 data for a set of tiles.
    Inputs:
        - tiles: a list of DLTile objects
        - start_date: the start date of the data
        - end_date: the end date of the data
        - clear_threshold: the threshold for cloud cover
        - batch_size: the number of tiles to process in each batch
    Outputs:
        - pred_gdf: a geodataframe of predictions
    """
    # Harmonized Sentinel-2 Level 2A collection.
    s2 = ee.ImageCollection('COPERNICUS/S2_HARMONIZED')

    # Cloud Score+ image collection. Note Cloud Score+ is produced from Sentinel-2
    # Level 1C data and can be applied to either L1C or L2A collections.
    csPlus = ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED')
    QA_BAND = 'cs_cdf'

    # Make a clear median composite.
    composite = (s2
        .filterDate(start_date, end_date)
        .linkCollection(csPlus, [QA_BAND])
        .map(lambda img: img.updateMask(img.select(QA_BAND).gte(clear_threshold)))
        .median())
    
    predictions = gpd.GeoDataFrame()
    completed_tasks = 0

    with concurrent.futures.ThreadPoolExecutor() as executor:
        for i in range(0, len(tiles), batch_size):
            batch_tiles = tiles[i:i+batch_size]
            
            # Process each tile in parallel
            futures = [executor.submit(process_tile, tile, composite) for tile in batch_tiles]
            
            # Collect the results as they become available
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                pixels, tile = result
                pred_gdf = predict_on_tile(tile, pixels, model, pred_threshold)
                predictions = pd.concat([predictions, pred_gdf], ignore_index=True)
                completed_tasks += 1
                print(f"Completed tasks: {completed_tasks}/{len(tiles)}", end='\r')
                # write data to a tmp file every 500 tiles
                if completed_tasks % 500 == 0:
                    predictions.to_file('tmp.geojson', driver='GeoJSON')
                # clear memory
                del pixels
                gc.collect()  # Perform garbage collection to free up memory
    
    return predictions

def process_tile(tile, composite):
    """
    Download Sentinel-2 data for a tile.
    Inputs:
        - tile: a DLTile object
        - composite: a Sentinel-2 image collection
    Outputs:
        - pixels: a numpy array containing the Sentinel-2 data
    """
    tile_geom = ee.Geometry.Rectangle(tile.geometry.bounds)
    composite_tile = composite.clipToBoundsAndScale(
        geometry=tile_geom,
        width=tile.tilesize + 2,
        height=tile.tilesize + 2)
    pixels = ee.data.computePixels({
        'bandIds': ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8A', 'B8', 'B9', 'B11', 'B12'],
        'expression': composite_tile,
        'fileFormat': 'NUMPY_NDARRAY',
        'grid': {'crsCode': tile.crs},
    })
    
    return pixels, tile

def predict_on_tile(tile_info, tile_data, model, pred_threshold=0.5):
    """
    Takes in a tile of data and a model
    Outputs a gdf of predictions and geometries
    """
    pixels = np.array(pad_patch(tile_data, tile_info.tilesize))
    pixels_norm = np.clip(pixels / 10000.0, 0, 1)
    chip_size = model.layers[0].input_shape[1]
    stride = chip_size // 2
    chips, chip_geoms = chips_from_tile(pixels_norm, tile_info, chip_size, stride)
    chip_geoms.to_crs('EPSG:4326', inplace=True)
    chips = np.array(chips)
    # make predictions on these chips as they come in
    # Can't hold all the chips in memory at once
    preds = model.predict(chips, verbose=0)
    # find the indices of the chips that are above the threshold
    pred_idx = np.where(preds > pred_threshold)[0]
    if len(pred_idx) > 0:
        # select the elements from the chip_geoms dataframe where the predictions are above the threshold
        preds_gdf = gpd.GeoDataFrame(geometry=chip_geoms['geometry'][pred_idx], crs='EPSG:4326')
        # add the predictions to the dataframe
        preds_gdf['pred'] = preds[pred_idx]
    else:
        preds_gdf = gpd.GeoDataFrame()

    return preds_gdf

def pad_patch(patch, height, width=None):
    """
    Depending on how a polygon falls across pixel boundaries, the resulting patch can be slightly
    bigger or smaller than intended.
    pad_patch trims pixels extending beyond the desired number of pixels if the
    patch is larger than desired. If the patch is smaller, it will fill the
    edge by reflecting the values.
    If trimmed, the patch should be trimmed from the center of the patch.
    Inputs:
        - patch: a numpy array of the shape the model requires
        - height: the desired height of the patch
        - width (optional): the desired width of the patch
    Outputs:
        - padded_patch: a numpy array of the desired shape
    """
    if width is None:
        width = height
    
    # convert from a structured array to a numpy array
    patch = np.array(patch.tolist())
    patch_height, patch_width,_ = patch.shape
    
    if patch_height > height:
        trim_top = (patch_height - height) // 2
        trim_bottom = trim_top + height
    else:
        trim_top = 0
        trim_bottom = patch_height
    
    if patch_width > width:
        trim_left = (patch_width - width) // 2
        trim_right = trim_left + width
    else:
        trim_left = 0
        trim_right = patch_width
    
    trimmed_patch = patch[trim_top:trim_bottom, trim_left:trim_right, :]
    
    if patch_height < height:
        pad_top = (height - patch_height) // 2
        pad_bottom = pad_top + patch_height
        padded_patch = np.pad(trimmed_patch, ((pad_top, height - pad_bottom), (0, 0), (0, 0)), mode='reflect')
    else:
        padded_patch = trimmed_patch
    
    if patch_width < width:
        pad_left = (width - patch_width) // 2
        pad_right = pad_left + patch_width
        padded_patch = np.pad(padded_patch, ((0, 0), (pad_left, width - pad_right), (0, 0)), mode='reflect')

    return padded_patch

def chips_from_tile(data, tile, width, stride):
    """
    Break a larger tile of Sentinel data into a set of patches that
    a model can process.
    Inputs:
        - data: Sentinel data. Typically a numpy masked array
        - tile_coords: bounds of the tile in the format (west, south, east, north)
        - stride: number of pixels between each patch
    Outputs:
        - chips: A list of numpy arrays of the shape the model requires
        - chip_coords: A geodataframe of the polygons corresponding to each chip
    """
    (west, south, east, north) = tile.bounds
    delta_x = east - west
    delta_y = south - north
    x_per_pixel = delta_x / np.shape(data)[0]

    y_per_pixel = delta_y / np.shape(data)[1]

    # The tile is broken into the number of whole patches
    # Regions extending beyond will not be padded and processed
    chip_coords = []
    chips = []

    # Extract patches and create a shapely polygon for each patch
    for i in range(0, np.shape(data)[0] - width + stride, stride):
        for j in range(0, np.shape(data)[1] - width + stride, stride):
            patch = data[j : j + width, i : i + width]
            chips.append(patch)

            nw_coord = [west + i * x_per_pixel, north + j * y_per_pixel]
            ne_coord = [west + (i + width) * x_per_pixel, north + j * y_per_pixel]
            sw_coord = [west + i * x_per_pixel, north + (j + width) * y_per_pixel]
            se_coord = [west + (i + width) * x_per_pixel, north + (j + width) * y_per_pixel]
            tile_geometry = [nw_coord, sw_coord, se_coord, ne_coord, nw_coord]
            chip_coords.append(shapely.geometry.Polygon(tile_geometry))
    chip_coords = gpd.GeoDataFrame(geometry=chip_coords, crs=tile.crs)
    return chips, chip_coords

def unit_norm(samples):
    """
    Channel-wise normalization of pixels in a patch.
    Means and deviations are constants generated from an earlier dataset.
    If changed, models will need to be retrained
    Input: (n,n,12) numpy array or list.
    Returns: normalized numpy array
    """
    means = [1405.8951, 1175.9235, 1172.4902, 1091.9574, 1321.1304, 2181.5363, 2670.2361, 2491.2354, 2948.3846, 420.1552, 2028.0025, 1076.2417]
    deviations = [291.9438, 398.5558, 504.557, 748.6153, 651.8549, 730.9811, 913.6062, 893.9428, 1055.297, 225.2153, 970.1915, 752.8637]
    normalized_samples = np.zeros_like(samples).astype('float32')
    for i in range(0, 12):
        #normalize each channel to global unit norm
        normalized_samples[:,:,i] = (np.array(samples.astype(float))[:,:,i] - means[i]) / deviations[i]
    return normalized_samples