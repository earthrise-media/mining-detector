from datetime import datetime, timedelta
import pickle
import os

from descarteslabs.geo import DLTile
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import shapely
from tqdm import tqdm

import gee
import utils


class Tile:
    """
    This is a simplification of a DLTIle to keep the interface consistent,
    but to allow for more specific definitions of tile geographies rather than the
    nearest dltile.
    inputs:
        lat: latitude
        lon: longitude
        resolution: resolution of the tile in meters
        tilesize: size of the tile in pixels
    outputs:
        tile: a DLTile like object
    """

    def __init__(self, lat, lon, tilesize):
        self.lat = lat
        self.lon = lon
        self.tilesize = tilesize
        self.resolution = 10

    def get_mgrs_crs(self):
        crs = DLTile.from_latlon(
            self.lat,
            self.lon,
            resolution=self.resolution,
            tilesize=self.tilesize,
            pad=0,
        ).crs
        self.crs = crs
        return crs

    def convert_to_mgrs(self):
        "converts the lat and lon from epsg:4326 to the mgrs crs"
        mgrs_crs = self.get_mgrs_crs()
        point = shapely.geometry.Point(self.lon, self.lat)
        gdf = gpd.GeoDataFrame(geometry=[point], crs="epsg:4326")
        gdf = gdf.to_crs(mgrs_crs)
        self.x = gdf.geometry.x[0]
        self.y = gdf.geometry.y[0]

    def create_geometry(self):
        "Creates a shapely geometry for the tile. Centered on the lat, lon, and extending out to the tilesize"
        self.convert_to_mgrs()
        center_point = shapely.geometry.Point(self.x, self.y)
        # I don't know why it works better when adding one to the tilesize
        buffer_distance = (
            (self.tilesize + 1) * 10 / 2
        )  # assume that resolution is always 10 because S2 data
        circle = center_point.buffer(buffer_distance)
        minx, miny, maxx, maxy = circle.bounds
        bbox = shapely.geometry.box(minx, miny, maxx, maxy)
        # convert from mgrs crs to epsg:4326
        bbox = gpd.GeoDataFrame(geometry=[bbox], crs=self.crs)
        bbox = bbox.to_crs("epsg:4326")
        bbox = bbox.geometry[0]
        self.geometry = bbox


class TrainingData:
    def __init__(self, patch_size, clear_threshold=0.75):
        self.patch_size = patch_size
        self.clear_threshold = clear_threshold
        
    def create_tiles(self, sampling_pts):

        sampling_pts.set_crs(epsg=4326, inplace=True)
        lats = sampling_pts.geometry.y
        lons = sampling_pts.geometry.x
        
        tiles = []
        for lat, lon in zip(lats, lons):
            tile = Tile(lat, lon, self.patch_size)
            tile.create_geometry()
            tiles.append(tile)
        print(f"{len(tiles)} tiles to download")
        return tiles

    def get_patches(self, path, label_class, date_col='date'):
        # get data for each tile

        sampling_pts = gpd.read_file(path)
        dates = sampling_pts[date_col].unique()
        
        for date in tqdm(dates):
            print(date)
            tiles = self.create_tiles(
                sampling_pts[sampling_pts[date_col] == date])

            dt = datetime.fromisoformat(date)
            start = (dt - timedelta(days=1)).isoformat()
            end = (dt + timedelta(days=1)).isoformat()

            s2_data = gee.GEE_Data_Extractor(
                tiles, start, end, clear_threshold=self.clear_threshold,
                collection='S2L2A')
            data, tiles = s2_data.get_patches()
            data = np.array(
                [utils.pad_patch(patch, self.patch_size) for patch in data])
            print(f"Retrieved {data.shape[0]} patches")

            outpath = f"{path.split('.geojson')[0]}{self.patch_size}_px{date}"
            save_patch_arrays(data, outpath, label_class)
            fig = utils.plot_numpy_grid(
                np.clip(data[:, :, :, (3, 2, 1)] / 3000, 0, 1))
            fig.savefig(f"{outpath}.png", bbox_inches="tight")



def save_patch_arrays(data, basepath, label_class):
    with open(basepath + "_patch_arrays.pkl", "wb") as f:
        pickle.dump(data, f)
    with open(basepath + "_patch_array_labels.pkl", "wb") as f:
        pickle.dump([label_class] * len(data), f)
