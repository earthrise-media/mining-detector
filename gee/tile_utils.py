
from affine import Affine
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Point, Polygon, box

from descarteslabs.geo import DLTile


class CenteredTile:
    """
    DLTile-like tile centered on a given lat/lon, not snapped to DL global grid.

    Attributes match DLTile where possible (key, crs, bounds, geotrans, shape).
    """

    def __init__(self, lat, lon, tilesize=48, resolution=10.0):
        self.lat = float(lat)
        self.lon = float(lon)
        self.tilesize = int(tilesize)
        self.resolution = float(resolution)

        snap_tile = DLTile.from_latlon(
            self.lat, self.lon,
            resolution=self.resolution,
            tilesize=self.tilesize,
            pad=0
        )
        self.crs = snap_tile.crs

        point = gpd.GeoSeries(
            [Point(self.lon, self.lat)], crs="EPSG:4326").to_crs(self.crs)
        x_center, y_center = float(point.x.iloc[0]), float(point.y.iloc[0])

        half_m = (self.tilesize * self.resolution) / 2.0
        raw_minx = x_center - half_m
        raw_maxx = x_center + half_m
        raw_miny = y_center - half_m
        raw_maxy = y_center + half_m

        ulx = np.floor(raw_minx / self.resolution) * self.resolution
        uly = np.ceil(raw_maxy / self.resolution) * self.resolution

        self.geotrans = Affine.translation(ulx, uly) * Affine.scale(
            self.resolution, -self.resolution)

        width = height = self.tilesize
        minx = ulx
        maxx = ulx + width * self.resolution
        maxy = uly
        miny = uly - height * self.resolution
        self.bounds = (minx, miny, maxx, maxy)
        
        self.shape = (self.tilesize, self.tilesize)
        self.key = (f"custom_{round(self.lat, 6)}_{round(self.lon, 6)}" +
                    f"_{self.resolution:.1f}_{self.tilesize}px")

        bbox_proj = box(minx, miny, maxx, maxy)
        self.geometry = gpd.GeoSeries(
            [bbox_proj], crs=self.crs).to_crs("EPSG:4326").iloc[0]

    def __repr__(self):
        return f"<CenteredTile key={self.key} res={self.resolution} size={self.tilesize}>"


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

def chips_from_tile(pixels, tile_info, chip_size, stride):
    (west, south, east, north) = tile_info.bounds
    delta_x = east - west
    delta_y = north - south  

    x_per_pixel = delta_x / pixels.shape[1]  
    y_per_pixel = delta_y / pixels.shape[0]  

    chips = []
    chip_coords = []

    for i in range(0, pixels.shape[1] - chip_size + stride, stride): 
        for j in range(0, pixels.shape[0] - chip_size + stride, stride): 
            patch = pixels[j : j + chip_size, i : i + chip_size]
            chips.append(patch)

            nw = (west + i * x_per_pixel, north - j * y_per_pixel)
            ne = (west + (i + chip_size) * x_per_pixel, north - j * y_per_pixel)
            sw = (west + i * x_per_pixel, north - (j + chip_size) * y_per_pixel)
            se = (west + (i + chip_size) * x_per_pixel,
                  north - (j + chip_size) * y_per_pixel)

            chip_coords.append(Polygon([nw, sw, se, ne, nw]))

    chip_coords = gpd.GeoDataFrame(geometry=chip_coords, crs=tile_info.crs)
    return chips, chip_coords

