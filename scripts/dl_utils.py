
import datetime
import json
import os

import descarteslabs as dl
from dateutil.relativedelta import relativedelta
import numpy as np
from scipy.stats import mode
import shapely
from tensorflow import keras

SENTINEL_BANDS = ['coastal-aerosol',
                  'blue',
                  'green',
                  'red',
                  'red-edge',
                  'red-edge-2',
                  'red-edge-3',
                  'nir',
                  'red-edge-4',
                  'water-vapor',
                  'swir1',
                  'swir2']

NORMALIZATION = 3000

def rect_from_point(coord, rect_height):
    """
    Create a geojson polygon from a coordinate pair.
    Inputs:
        - coord: coordinates in the form [lon, lat]
        - rect_height: the height of the rectangle in degrees latitude
    Returns: Geojson formatted polygon
    """
    lon, lat = coord
    lat_w = rect_height / 2
    lon_w = lat_w / np.cos(np.deg2rad(lat))
    rect = shapely.geometry.mapping(shapely.geometry.box(
        lon - lon_w, lat - lat_w, lon + lon_w, lat + lat_w))
    return rect

def get_tiles_from_roi(roi_file, tilesize, pad):
    """Retrieve tile keys covering ROI."""
    with open(roi_file, 'r') as f:
        fc = json.load(f)
        try:
            features = fc['features']
        except KeyError:
            features = fc['geometries']

    all_keys = list()
    ctr =0
    for feature in features:
        tiles = dl.Raster().iter_dltiles_from_shape(10.0, tilesize, pad,
                                                    feature)
        for tile in tiles:
            all_keys.append(tile['properties']['key'])
            ctr +=1
            print(ctr, end='\r')

    print('Split ROI into {} tiles'.format(len(all_keys)))
    return all_keys

def download_patch(polygon, start_date, end_date, s2_id='sentinel-2:L1C',
                   s2cloud_id='sentinel-2:L1C:dlcloud:v1'):
    """
    Download a stack of cloud-masked Sentinel data
    Inputs:
        - polygon: Geojson polygon enclosing the region of data to be extracted
        - start_date/end_date: The time bounds of the search
    Returns:
        - A list of images of shape (height, width, channels)
    """
    cloud_scenes, _ = dl.scenes.search(
        polygon,
        products=[s2cloud_id],
        start_datetime=start_date,
        end_datetime=end_date,
        limit=None
    )

    scenes, geoctx = dl.scenes.search(
        polygon,
        products=[s2_id],
        start_datetime=start_date,
        end_datetime=end_date,
        limit=None
    )

    # select only scenes that have a cloud mask
    cloud_dates = [scene.properties.acquired for scene in cloud_scenes]
    dates = [scene.properties.acquired for scene in scenes]
    shared_dates = set(cloud_dates) & set(dates)
    scenes = scenes.filter(
        lambda x: x.properties.acquired in shared_dates)
    cloud_scenes = cloud_scenes.filter(
        lambda x: x.properties.acquired in shared_dates)

    # A cloud stack is an array with shape (num_img, data_band, height, width)
    # A value of 255 means that the pixel is cloud free, 0 means cloudy
    cloud_stack = cloud_scenes.stack(bands=['valid_cloudfree'], ctx=geoctx)

    img_stack, raster_info = scenes.stack(
        bands=SENTINEL_BANDS, ctx=geoctx, raster_info=True)
    cloud_masks = np.repeat(cloud_stack, repeats = 12, axis=1)

    # Add cloud masked pixels to the image mask
    img_stack.mask[cloud_masks.data == 0] = True

    # Remove fully masked images and reorder to channels last
    # TODO: remove raster infor for fully masked images too
    img_stack = [np.moveaxis(img, 0, -1) for img in img_stack
                     if np.sum(img) > 0]

    return img_stack, raster_info

def pad_patch(patch, height, width=None):
    """
    Depending on how a polygon falls across pixel boundaries, it can be slightly
    bigger or smaller than intended.
    pad_patch trims pixels extending beyond the desired number of pixels if the
    patch is larger than desired. If the patch is smaller, it will fill the
    edge by reflecting the values.
    """
    h, w, c = patch.shape
    if width:
        if h < height:
            patch = np.pad(patch, (height - h, 0), mode='reflect')
        if w < width:
            patch = np.pad(patch, (0, width - w), mode='reflect')
        patch = patch[:height, :width, :12]
    else:
        if h < height or w < height:
            print(width)
            print(np.min([h, w]))
            patch = np.pad(patch, height - np.min([h, w]), mode='reflect')
        patch = patch[:height, :height, :12]
    return patch

def trim_patch(patch, height, width=None):
    """
    Depending on how a polygon falls across pixel boundaries, it can be slightly
    bigger or smaller than intended.
    pad_patch trims pixels extending beyond the desired number of pixels if the
    patch is larger than desired. If the patch is smaller, it will fill the
    edge by reflecting the values.
    """
    h, w, c = patch.shape
    margin = int(np.floor((h - height) / 2))
    trimmed = patch[margin:margin+height, margin:margin+height]
    return trimmed

def download_batches(polygon, start_date, end_date, batch_months):
    """Download cloud-masked Sentinel imagery in time-interval batches.

    Args:
        polygon: A GeoJSON-like polygon
        start_date: Isoformat start date
        end_date: Isoformat end start
        batch_months: Batch length in integer number of months

    Returns: List of lists of images, one list per batch
    """
    batches, raster_infos = [], []
    delta = relativedelta(months=batch_months)
    start = datetime.date.fromisoformat(start_date)
    end = start + delta
    while end <= datetime.date.fromisoformat(end_date):
        try:
            batch, raster_info = download_patch(polygon, start.isoformat(),
                                                end.isoformat())
        except IndexError as e:
            print(f'Failed to retreive month {start.isoformat()}: {repr(e)}')
            batch, raster_info = [], []
        # Sometimes there are patches with no data. Ignore those
        if len(np.shape(batch)) > 1:
            batches.append(batch)
            raster_infos.append(raster_info)
        start += delta
        end += delta
    return batches, raster_infos

def download_mosaics(polygon, start_date, end_date, mosaic_period=1,
                     method='median'):
    """Download cloud-masked Sentinel image mosaics

    Args:
        polygon: A GeoJSON-like polygon
        start_date: Isoformat start date
        end_date: Isoformat end start
        mosaic_period: Integer months over which to mosaic image data
        method: String method to pass to mosaic()

    Returns: List of image mosaics and list of meta-datas
    """
    batches, raster_infos = download_batches(polygon, start_date, end_date,
                                                 mosaic_period)
    mosaics = [mosaic(batch, method) for batch in batches]
    # There are cases where some patches are sized differently
    # If that is the case, pad/clip them to the same shape
    heights = [np.shape(img)[0] for img in mosaics]
    widths = [np.shape(img)[1] for img in mosaics]
    if len(np.unique(heights)) > 1 or len(np.unique(widths)) > 1:
        h = mode(heights).mode[0]
        w = mode(widths).mode[0]
        mosaics = [np.ma.masked_array(pad_patch(img.data, h, w),
                                        pad_patch(img.mask, h, w)) for img in mosaics]
    mosaic_info = [next(iter(r)) for r in raster_infos]
    return mosaics, mosaic_info

def mosaic(arrays, method):
    """Mosaic masked arrays.

    Args:
        arrays: A list of masked arrays
        method:
            'median': return the median of valid pixel values
            'min': return the minimum of valid pixel values
            'min_masked': return the array with fewest masked pixels

    Returns: A masked array or None if arrays is an empty list
    """
    if not arrays:
        return

    if method == 'median':
        stack = np.ma.stack(arrays)
        reduced = np.ma.median(stack, axis=0)
    elif method == 'min':
        stack = np.ma.stack(arrays)
        reduced = np.ma.min(stack, axis=0)
    elif method == 'min_masked':
        mask_sorted = sorted(arrays, key=lambda p:np.sum(p.mask))
        reduced = next(iter(mask_sorted))
    else:
        raise ValueError(f'Method {method} not recognized.')

    return reduced

def normalize(x):
    return np.array(x) / NORMALIZATION

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
        normalized_samples[:,:,i] = (np.array(samples)[:,:,i] - means[i]) / deviations[i]
    return normalized_samples

def patches_from_tile(tile, raster_info, width, stride):
    """
    Break a larger tile of Sentinel data into a set of patches that
    a model can process.
    Inputs:
        - tile: Sentinel data. Typically a numpy masked array
        - raster_info: Descartes metadata for the tile
        - model: keras model
        - stride: number of pixels between each patch
    Outputs:
        - patches: A list of numpy arrays of the shape the model requires
        - patch_coords: A list of shapely polygon features describing the patch bounds
    """
    patch_coords = raster_info[0]['wgs84Extent']['coordinates'][0]
    delta_lon = patch_coords[2][0] - patch_coords[0][0]
    delta_lat = patch_coords[1][1] - patch_coords[0][1]
    lon_degrees_per_pixel = delta_lon / np.shape(tile)[0]
    lat_degrees_per_pixel = delta_lat / np.shape(tile)[1]
    top_left = patch_coords[0]

    # The tile is broken into the number of whole patches
    # Regions extending beyond will not be padded and processed
    patch_coords = []
    patches = []

    # Extract patches and create a shapely polygon for each patch
    for i in range(0, np.shape(tile)[0] - width, stride):
        for j in range(0, np.shape(tile)[1] - width, stride):
            patch = tile[j : j + width,
                         i : i + width]
            patches.append(patch)

            nw_coord = [top_left[0] + i * lon_degrees_per_pixel,
                        top_left[1] + j * lat_degrees_per_pixel]
            ne_coord = [top_left[0] + (i + width) * lon_degrees_per_pixel,
                        top_left[1] + j * lat_degrees_per_pixel]
            sw_coord = [top_left[0] + i * lon_degrees_per_pixel,
                        top_left[1] + (j + width) * lat_degrees_per_pixel]
            se_coord = [top_left[0] + (i + width) * lon_degrees_per_pixel,
                        top_left[1] + (j + width) * lat_degrees_per_pixel]
            tile_geometry = [nw_coord, sw_coord, se_coord, ne_coord, nw_coord]
            patch_coords.append(shapely.geometry.Polygon(tile_geometry))
    return patches, patch_coords

class DescartesRun(object):
    """Class to manage bulk model prediction on the Descartes Labs platform.

    Attributes:
        patch_product_id: DL id for output rasters
        product_name: String identifer for output rasters
        product: Instantiated dl.catalog.Product
        patch_model_name: String identifier for learned Keras model
        model: Instantiated Keras model
        mosaic_period: Integer number of months worth of data to mosaic
        mosaic_method: Compositing method for the mosaic() function
        input_bands: List of DL names identifying Sentinel bands

    External methods:
        init_product: Create or get DL catalog product with specified bands.
        upload_patch_model: Upload model to DL storage.
        init_patch_model: Instantiate model from DL storage.
        __call__: Run model on a geographic tile.
    """
    def __init__(self,
                 patch_product_id,
                 product_name='',
                 patch_model_file='',
                 patch_model_name='',
                 patch_stride=None,
                 mosaic_period=1,
                 mosaic_method='min',
                 nodata=-1,
                 input_bands=SENTINEL_BANDS,
                 **kwargs):

        if patch_product_id.startswith('earthrise:'):
            self.patch_product_id = patch_product_id
        else:
            self.patch_product_id = f'earthrise:{patch_product_id}'
        self.product_name = product_name if product_name else self.product_id
        self.nodata = nodata

        self.patch_model_name = patch_model_name
        self.upload_patch_model(patch_model_file)
        self.patch_model = self.init_patch_model()
        self.patch_product = self.init_patch_product()
        if patch_stride:
            self.patch_stride = patch_stride
        else:
            self.patch_stride = self.patch_model.input_shape[1]

        self.mosaic_period = mosaic_period
        self.mosaic_method = mosaic_method

        self.input_bands = input_bands

    def init_patch_product(self):
        """Create or get DL catalog product."""
        fc_ids = [fc.id for fc in dl.vectors.FeatureCollection.list()]
        product_id = None
        for fc in fc_ids:
            if self.patch_product_id in fc:
                product_id = fc

        if not product_id:
            print("Creating product", self.patch_product_id + '_patches')
            product = dl.vectors.FeatureCollection.create(product_id=self.patch_product_id + '_patches',
                                                          title=self.product_name + '_patches',
                                                          description=self.patch_model_name)
        else:
            print(f"Product {self.patch_product_id}_patches already exists...")
            product = dl.vectors.FeatureCollection(product_id)
        return product


    def upload_patch_model(self, patch_model_file):
        """Upload model to DL storage."""
        if dl.Storage().exists(self.patch_model_name):
            print(f'Model {self.patch_model_name} found in DLStorage.')
        else:
            dl.Storage().set_file(self.patch_model_name, patch_model_file)
            print(f'Model {patch_model_file} uploaded with key {self.patch_model_name}.')

    def init_patch_model(self):
        """Instantiate model from DL storage."""
        temp_file = 'tmp-' + self.patch_model_name
        dl.Storage().get_file(self.patch_model_name, temp_file)
        patch_model = keras.models.load_model(temp_file, custom_objects={'LeakyReLU': keras.layers.LeakyReLU,
                                                                         'ELU': keras.layers.ELU,
                                                                         'ReLU': keras.layers.ReLU
                                                                         })
        os.remove(temp_file)
        return patch_model

    def __call__(self, dlkey, start_date, end_date):
        """Run model on a geographic tile.

        Args:
            dlkey: Key idenifying a DL tile.
            start_date: Isoformat begin date for prediction window.
            end_date: Isoformat end date for prediction window.

        Returns: None. (Uploads raster output to DL storage.)
        """
        tile = dl.scenes.DLTile.from_key(dlkey)
        mosaics, raster_info = download_mosaics(
            tile, start_date, end_date, self.mosaic_period, self.mosaic_method)

        # Spatial patch classifier prediction

        # Generate a list of coordinates for the patches within the tile
        _, patch_coords = patches_from_tile(mosaics[0], raster_info, self.patch_model.input_shape[2], self.patch_stride)

        # Initialize a dictionary where the patch coordinate boundaries are the keys
        # Each value is an empty list where predictions will be appended
        pred_dict = {tuple(coord.bounds): [] for coord in patch_coords}

        # Set a threshold for acceptable cloudiness within a patch for a prediction to be valid
        patch_cloud_threshold = 0.1
        input_h = self.patch_model.input_shape[1]
        input_w = self.patch_model.input_shape[2]

        for image in mosaics:
            # generate patches for first image in pair
            print("image shape", image.shape)
            patches, _ = patches_from_tile(image, raster_info, input_h, self.patch_stride)

            patch_stack = []
            cloud_free = []
            for patch in patches:
                #Note: Should this be converted to trim patch?
                model_input = pad_patch(patch.filled(0), input_h, input_w)

                # if using basic normalization: patch_stack.append(np.clip(normalize(model_input), 0, 1))
                patch_stack.append(unit_norm(model_input))

                # Evaluate whether both patches in a sample are below cloud limit
                cloudiness = np.sum(patch.mask) / np.size(patch.mask)
                if cloudiness < patch_cloud_threshold:
                    cloud_free.append(True)
                else:
                    cloud_free.append(False)

            preds = self.patch_model.predict(np.array(patch_stack))[:,1]

            # If patches were cloud free, append the prediction to the dictionary
            for coord, pred, cloud_bool in zip(patch_coords, preds, cloud_free):
                if  cloud_bool == True:
                    pred_dict[tuple(coord.bounds)].append(pred)

        # Create dictionary for outputs.
        feature_list = []
        for coords, key in zip(patch_coords, pred_dict):
            preds = [round(pred, 4) for pred in pred_dict[key]]
            geometry = shapely.geometry.mapping(coords)
            if len(preds) > 0:
                properties = {
                    'mean': np.mean(preds, axis=0).astype('float'),
                    'median': np.median(preds, axis=0).astype('float'),
                    'min': np.min(preds, axis=0).astype('float'),
                    'max': np.max(preds, axis=0).astype('float'),
                    'std': np.std(preds, axis=0).astype('float'),
                    'count': np.shape(preds)[0],
                }
            else:
                properties = {
                    'mean': -1,
                    'median': -1,
                    'min': -1,
                    'max': -1,
                    'std': -1,
                    'count': np.shape(preds)[0],
                }
            # To save output space, only save outputs that are above a threshold
            # TODO: This value should be a command-line argument
            if properties['mean'] > 0.5:
                feature_list.append(dl.vectors.Feature(geometry = geometry, properties = properties))
        print(len(feature_list), 'features generated')
        if len(feature_list) > 0:
            self.patch_product.add(feature_list)
