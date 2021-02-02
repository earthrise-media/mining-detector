import ee
import geemap
from multiprocessing.dummy import Pool as ThreadPool
from tqdm import tqdm
from datetime import datetime
from functools import partial
import json
import os
import numpy as np
import pandas as pd

ee.Initialize()

# Sentinel 2 band descriptions
band_descriptions = {
    'B1': 'Aerosols, 442nm',
    'B2': 'Blue, 492nm',
    'B3': 'Green, 559nm',
    'B4': 'Red, 665nm',
    'B5': 'Red Edge 1, 704nm',
    'B6': 'Red Edge 2, 739nm',
    'B7': 'Red Edge 3, 779nm',
    'B8': 'NIR, 833nm',
    'B8A': 'Red Edge 4, 864nm',
    'B9': 'Water Vapor, 943nm',
    'B11': 'SWIR 1, 1610nm',
    'B12': 'SWIR 2, 2186nm'
}

band_wavelengths = [442, 492, 559, 665, 704, 739, 779, 833, 864, 943, 1610, 2186]


## S2 Cloud Filtering Functions

CLOUD_FILTER = 30
CLD_PRB_THRESH = 30
NIR_DRK_THRESH = 0.15
CLD_PRJ_DIST = 1
BUFFER = 50
#DATASET = 'COPERNICUS/S2_SR'
DATASET = 'COPERNICUS/S2'

def get_s2_sr_cld_col(aoi, start_date, end_date):
    """
    Creates an ImageCollection for a region and time period.
    ImageCollection is prefiltered by the QA60 cloud mask band
    Prefiltering percentage specified by global `CLOUD_FILTER` variable
    """
    # Import and filter S2 SR.
    s2_sr_col = (ee.ImageCollection(DATASET)
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', CLOUD_FILTER)))

    # Import and filter s2cloudless.
    s2_cloudless_col = (ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')
        .filterBounds(aoi)
        .filterDate(start_date, end_date))

    # Join the filtered s2cloudless collection to the SR collection by the 'system:index' property.
    return ee.ImageCollection(ee.Join.saveFirst('s2cloudless').apply(**{
        'primary': s2_sr_col,
        'secondary': s2_cloudless_col,
        'condition': ee.Filter.equals(**{
            'leftField': 'system:index',
            'rightField': 'system:index'
        })
    }))

def add_cloud_bands(img):
    """
    From the s2_cloud_probability dataset, return an image with
    cloud probabilities below the global `CLD_PRB_THRESH` variable
    """
    # Get s2cloudless image, subset the probability band.
    cld_prb = ee.Image(img.get('s2cloudless')).select('probability')

    # Condition s2cloudless by the probability threshold value.
    is_cloud = cld_prb.gt(CLD_PRB_THRESH).rename('clouds')

    # Add the cloud probability layer and cloud mask as image bands.
    return img.addBands(ee.Image([cld_prb, is_cloud]))

def add_shadow_bands(img):
    """
    Isolate cloud shadows over land
    Cloud shadow thresholds are given by the global `NIR_DRK_THRESH` variable
    CK Note: I don't think this algorithm works over water
    """

    SR_BAND_SCALE = 1e4
    # CK Note: Removing the not_water condition
    dark_pixels = img.select('B8').lt(NIR_DRK_THRESH*SR_BAND_SCALE).rename('dark_pixels')

    # Identify water pixels from the SCL band.
    #not_water = img.select('SCL').neq(6)

    # Identify dark NIR pixels that are not water (potential cloud shadow pixels).

    #dark_pixels = img.select('B8').lt(NIR_DRK_THRESH*SR_BAND_SCALE).multiply(not_water).rename('dark_pixels')



    # Determine the direction to project cloud shadow from clouds (assumes UTM projection).
    shadow_azimuth = ee.Number(90).subtract(ee.Number(img.get('MEAN_SOLAR_AZIMUTH_ANGLE')));

    # Project shadows from clouds for the distance specified by the CLD_PRJ_DIST input.
    cld_proj = (img.select('clouds').directionalDistanceTransform(shadow_azimuth, CLD_PRJ_DIST*10)
        .reproject(**{'crs': img.select(0).projection(), 'scale': 100})
        .select('distance')
        .mask()
        .rename('cloud_transform'))

    # Identify the intersection of dark pixels with cloud shadow projection.
    shadows = cld_proj.multiply(dark_pixels).rename('shadows')

    # Add dark pixels, cloud projection, and identified shadows as image bands.
    return img.addBands(ee.Image([dark_pixels, cld_proj, shadows]))

def add_cld_shdw_mask(img):
    """
    Create a mask based on the cloud and cloud shadow images
    """
    # Add cloud component bands.
    img_cloud = add_cloud_bands(img)

    # Add cloud shadow component bands.
    img_cloud_shadow = add_shadow_bands(img_cloud)

    # Combine cloud and shadow mask, set cloud and shadow as value 1, else 0.
    is_cld_shdw = img_cloud_shadow.select('clouds').add(img_cloud_shadow.select('shadows')).gt(0)

    # Remove small cloud-shadow patches and dilate remaining pixels by BUFFER input.
    # 20 m scale is for speed, and assumes clouds don't require 10 m precision.
    is_cld_shdw = (is_cld_shdw.focal_min(2).focal_max(BUFFER*2/20)
        .reproject(**{'crs': img.select([0]).projection(), 'scale': 20})
        .rename('cloudmask'))

    # Add the final cloud-shadow mask to the image.
    return img_cloud_shadow.addBands(is_cld_shdw)

def apply_cld_shdw_mask(img):
    """
    Apply the cloud mask to the all Sentinel bands beginning with `B`
    """
    # Subset the cloudmask band and invert it so clouds/shadow are 0, else 1.
    not_cld_shdw = img.select('cloudmask').Not()

    # Subset reflectance bands and update their masks, return the result.
    return img.select('B.*').updateMask(not_cld_shdw)


def create_rect(lon, lat, width):
    """
    Given a set of coordinates, create an earth engine rect of a fixed width/height
    """
    extent = width / 2
    rect = ee.Geometry.Polygon([[lon + extent, lat + extent],
                                [lon + extent, lat - extent],
                                [lon - extent, lat - extent],
                                [lon - extent, lat + extent]], None, False)
    return rect

## Get Sentinel Data
def get_sentinel_band(site_name, roi, output_dict, image, scale, band):
    band_img = image.select(band).clipToBoundsAndScale(roi, scale=scale)
    image_array = geemap.ee_to_numpy(band_img, region=roi, default_value=-999)
    patch = np.squeeze(image_array)
    if patch.all() != None:
        output_dict[band] = np.squeeze(image_array)
    else:
        output_dict[band] = []
    return patch

def get_patches(site_names, site_coords, rect_width, image, scale):
    """
    Multithreaded process to export Sentinel 2 patches as numpy arrays.
    Input lists of site names and site coordinates along with an Earth Engine image.
    Exports each band in image to a dictionary organized by [site name][band][band_img]
    """
    patch_dict = {}
    for name, site in zip(site_names, site_coords):
        print("Downloading", name)
        pool = ThreadPool(12)
        roi = create_rect(site[0], site[1], rect_width)
        images = {}
        bands = list(band_descriptions.keys())
        get_sentinel_partial = partial(get_sentinel_band,
                                       name,
                                       roi,
                                       images,
                                       image,
                                       scale)
        pool.map(get_sentinel_partial, bands)
        pool.close()
        pool.join()
        patch_dict[name] = images
    return patch_dict

def get_tpa_patches(site_names, polygons, image):
    """
    Multithreaded process to export Sentinel 2 patches as numpy arrays.
    Input lists of site names and site coordinates along with an Earth Engine image.
    Exports each band in image to a dictionary organized by [site name][band][band_img]
    """
    patch_dict = {}
    for name, roi in zip(site_names, polygons):
        print("Processing", name)
        pool = ThreadPool(12)
        images = {}
        bands = list(band_descriptions.keys())
        get_sentinel_partial = partial(get_sentinel_band,
                                       name,
                                       roi,
                                       images,
                                       image)
        pool.map(get_sentinel_partial, bands)
        pool.close()
        pool.join()
        patch_dict[name] = images
    return patch_dict

def get_history(coords, name, width, num_months=22, start_date='2019-01-01', cloud_mask=True, scale=10):
    history = {}

    # TODO: This ROI is only set by the first coordinate pair with a huge
    # rect width. Would be great to find a bounding box around all coords.
    lons = [coord[0] for coord in coords]
    lats = [coord[1] for coord in coords]
    roi = ee.Geometry.Rectangle((np.min(lons), np.min(lats), np.max(lons), np.max(lats)))
    #roi = create_rect(np.mean([coord[0] for coord in coords]), np.mean([coord[1] for coord in coords]), 6.5)
    date = ee.Date(start_date)
    for month in tqdm(range(num_months)):
        s2_data = get_s2_sr_cld_col(roi, date, date.advance(1, 'month'))
        if cloud_mask:
            s2_sr_median = s2_data.map(add_cld_shdw_mask) \
                                    .map(apply_cld_shdw_mask) \
                                    .median()
        else:
            s2_sr_median = s2_data.median()

        patches = get_patches(name, coords, width, s2_sr_median, scale)
        date_text = str(datetime.fromtimestamp(date.getInfo()['value'] // 1000 + 86400).date())
        history[date_text] = patches
        date = date.advance(1, 'month')

    return history

def get_history_polygon(coords, name, polygons, width, num_months=22, start_date='2019-01-01', cloud_mask=True):
    history = {}

    # TODO: This ROI is only set by the first coordinate pair with a huge
    # rect width. Would be great to find a bounding box around all coords.
    roi = create_rect(coords[0][0], coords[0][1], 5.5)
    date = ee.Date(start_date)
    for month in tqdm(range(num_months)):
        s2_data = get_s2_sr_cld_col(roi, date, date.advance(1, 'month'))
        if cloud_mask:
            s2_sr_median = s2_data.map(add_cld_shdw_mask) \
                                    .map(apply_cld_shdw_mask) \
                                    .median()
        else:
            s2_sr_median = s2_data.median()

        patches = get_tpa_patches(name, polygons, s2_sr_median)
        date_text = str(datetime.fromtimestamp(date.getInfo()['value'] // 1000 + 86400).date())
        history[date_text] = patches
        date = date.advance(1, 'month')

    return history

def get_pixel_vectors(data_source, month):
    pixel_vectors = []
    width, height = 0, 0
    for site in data_source[list(data_source.keys())[0]]:
        if -999 not in data_source[month][site]['B2']:
            bands = data_source[month][site]
            if 0 not in [len(bands[band]) for band in bands]:
                width, height = np.shape(bands['B2'])
                for i in range(width):
                    for j in range(height):
                        pixel_vector = []
                        for band_name in band_descriptions:
                            pixel_vector.append(bands[band_name][i][j])
                        pixel_vectors.append(pixel_vector)
    return pixel_vectors, width, height
