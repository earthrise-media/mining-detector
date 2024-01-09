import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import shapely

from descarteslabs.geo import DLTile


def create_tiles(region, tilesize, padding, resolution=10):
    """
    Create a set of tiles that cover a region.
    Inputs:
        - region: a geojson polygon
        - tilesize: the size of the tiles in pixels
        - padding: the number of pixels to pad each tile
    Outputs:
        - tiles: a list of DLTile objects
    """
    tiles = DLTile.iter_from_shape(
        region, tilesize=tilesize, resolution=resolution, pad=padding
    )
    tiles = [tile for tile in tiles]
    return tiles


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

    patch_height, patch_width, _ = patch.shape

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
        padded_patch = np.pad(
            trimmed_patch,
            ((pad_top, height - pad_bottom), (0, 0), (0, 0)),
            mode="reflect",
        )
    else:
        padded_patch = trimmed_patch

    if patch_width < width:
        pad_left = (width - patch_width) // 2
        pad_right = pad_left + patch_width
        padded_patch = np.pad(
            padded_patch,
            ((0, 0), (pad_left, width - pad_right), (0, 0)),
            mode="reflect",
        )

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
            se_coord = [
                west + (i + width) * x_per_pixel,
                north + (j + width) * y_per_pixel,
            ]
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
    means = [
        1405.8951,
        1175.9235,
        1172.4902,
        1091.9574,
        1321.1304,
        2181.5363,
        2670.2361,
        2491.2354,
        2948.3846,
        420.1552,
        2028.0025,
        1076.2417,
    ]
    deviations = [
        291.9438,
        398.5558,
        504.557,
        748.6153,
        651.8549,
        730.9811,
        913.6062,
        893.9428,
        1055.297,
        225.2153,
        970.1915,
        752.8637,
    ]
    normalized_samples = np.zeros_like(samples).astype("float32")
    for i in range(0, 12):
        # normalize each channel to global unit norm
        normalized_samples[:, :, i] = (
            np.array(samples.astype(float))[:, :, i] - means[i]
        ) / deviations[i]
    return normalized_samples


def plot_numpy_grid(patches):
    num_img = int(np.ceil(np.sqrt(len(patches))))
    padding = 1
    h, w, c = patches[0].shape
    mosaic = np.zeros((num_img * (h + padding), num_img * (w + padding), c))
    counter = 0
    for i in range(num_img):
        for j in range(num_img):
            if counter < len(patches):
                mosaic[
                    i * (h + padding) : (i + 1) * h + i * padding,
                    j * (w + padding) : (j + 1) * w + j * padding,
                ] = patches[counter]
            else:
                mosaic[
                    i * (h + padding) : (i + 1) * h + i * padding,
                    j * (w + padding) : (j + 1) * w + j * padding,
                ] = np.zeros((h, w, c))
            counter += 1

    fig, ax = plt.subplots(figsize=(num_img, num_img), dpi=150)
    ax.axis("off")
    ax.imshow(mosaic)
    return fig
