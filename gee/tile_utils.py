
from affine import Affine
from descarteslabs.geo import DLTile
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Point, Polygon, box

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
        self.pad = 0

        snap_tile = DLTile.from_latlon(
            self.lat, self.lon,
            resolution=self.resolution,
            tilesize=self.tilesize,
            pad=self.pad
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


def create_tiles(region, tilesize, pad, resolution=10):
    """
    Create a set of tiles that cover a region.
    Inputs:
        - region: a geojson polygon
        - tilesize: the size of the tiles in pixels
        - pad: the number of pixels to pad each tile
        - resolution: spatial resolution in meters
    Outputs:
        - tiles: a list of DLTile objects
    """
    tiles = DLTile.iter_from_shape(
        region, tilesize=tilesize, resolution=resolution, pad=pad
    )
    return list(tiles)


def ensure_tile_shape(raster, height, width=None):
    """
    Ensure tile raster data has the exact requested shape by trimming 
    (from center) if too large or padding (reflect) if too small.

    Parameters
    ----------
    raster : np.ndarray
        Input raster with shape (H, W, C).
    height : int
        Desired height.
    width : int, optional
        Desired width. If None, equals `height`.

    Returns
    -------
    np.ndarray
        Raster of shape (height, width, C).
    """
    if width is None:
        width = height

    raster_height, raster_width, _ = raster.shape

    # --- Compute cropping (trim from center if too big) ---
    dh = max(raster_height - height, 0)
    dw = max(raster_width - width, 0)

    trim_top = dh // 2
    trim_bottom = raster_height - (dh - trim_top)
    trim_left = dw // 2
    trim_right = raster_width - (dw - trim_left)

    cropped = raster[trim_top:trim_bottom, trim_left:trim_right, :]

    # --- Compute padding (reflect if too small) ---
    ph = max(height - cropped.shape[0], 0)
    pw = max(width - cropped.shape[1], 0)

    pad_top = ph // 2
    pad_bottom = ph - pad_top
    pad_left = pw // 2
    pad_right = pw - pad_left

    padded = np.pad(
        cropped,
        ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
        mode="reflect"
    )

    return padded

def chips_from_tile(pixels, tile, chip_size, stride):
    (west, south, east, north) = tile.bounds
    delta_x = east - west
    delta_y = north - south  

    x_per_pixel = delta_x / pixels.shape[1]  
    y_per_pixel = delta_y / pixels.shape[0]  

    chips = []
    chip_coords = []

    for i in range(0, pixels.shape[1] - chip_size + stride, stride): 
        for j in range(0, pixels.shape[0] - chip_size + stride, stride): 
            chip = pixels[j : j + chip_size, i : i + chip_size]
            chips.append(chip)

            nw = (west + i * x_per_pixel, north - j * y_per_pixel)
            ne = (west + (i + chip_size) * x_per_pixel, north - j * y_per_pixel)
            sw = (west + i * x_per_pixel, north - (j + chip_size) * y_per_pixel)
            se = (west + (i + chip_size) * x_per_pixel,
                  north - (j + chip_size) * y_per_pixel)

            chip_coords.append(Polygon([nw, sw, se, ne, nw]))

    chip_coords = gpd.GeoDataFrame(geometry=chip_coords, crs=tile.crs)
    return chips, chip_coords

