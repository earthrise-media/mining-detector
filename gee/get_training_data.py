import pickle
import os

from descarteslabs.geo import DLTile
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import shapely

import gee_data

class Tile():
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
        crs = DLTile.from_latlon(self.lat, self.lon, resolution=self.resolution, tilesize=self.tilesize, pad=0).crs
        self.crs = crs
        return crs
    
    def convert_to_mgrs(self):
        "converts the lat and lon from epsg:4326 to the mgrs crs"
        mgrs_crs = self.get_mgrs_crs()
        point = shapely.geometry.Point(self.lon, self.lat)
        gdf = gpd.GeoDataFrame(geometry=[point], crs='epsg:4326')
        gdf = gdf.to_crs(mgrs_crs)
        self.x = gdf.geometry.x[0]
        self.y = gdf.geometry.y[0]

    
    def create_geometry(self):
        "Creates a shapely geometry for the tile. Centered on the lat, lon, and extending out to the tilesize"
        self.convert_to_mgrs()
        center_point = shapely.geometry.Point(self.x, self.y)
        buffer_distance = self.tilesize * 10 / 2 # assume that resolution is always 10 because S2 data
        circle = center_point.buffer(buffer_distance)
        minx, miny, maxx, maxy = circle.bounds
        bbox = shapely.geometry.box(minx, miny, maxx, maxy)
        # convert from mgrs crs to epsg:4326
        bbox = gpd.GeoDataFrame(geometry=[bbox], crs=self.crs)
        bbox = bbox.to_crs('epsg:4326')
        bbox = bbox.geometry[0]
        self.geometry = bbox

class TrainingData():

    def __init__(self, sampling_file, label_class, patch_size, start_date, end_date, clear_threshold=0.75):
        self.sampling_file = sampling_file
        self.label_class = label_class
        self.patch_size = patch_size
        self.start_date = start_date
        self.end_date = end_date
        self.clear_threshold = clear_threshold

    def create_tiles(self):
        self.sampling_locations = gpd.read_file(f'../data/sampling_locations/{self.sampling_file}.geojson')
        self.sampling_locations.set_crs(epsg=4326, inplace=True)
        lats = self.sampling_locations.geometry.y
        lons = self.sampling_locations.geometry.x
        # create a  for each sampling location
        tiles = []
        for lat, lon in zip(lats, lons):
            tile = Tile(lat, lon, self.patch_size)
            tile.create_geometry()
            tiles.append(tile)
        print(f"{len(tiles)} tiles to download")
        self.tiles = tiles

    def get_patches(self):
        # get data for each tile
        
        self.create_tiles()

        print(f'Getting data from {self.start_date} to {self.end_date}')
        data = gee_data.get_image_data(self.tiles, self.start_date, self.end_date, self.clear_threshold)
        data = np.array([gee_data.pad_patch(patch, self.patch_size) for patch in data])
        print(f'Retrieved {data.shape[0]} patches')

        

        # save the data
        basepath = f'../data/training_data/{self.patch_size}_px/'
        # create directory if it doesn't exist
        if not os.path.exists(basepath):
            os.makedirs(basepath)
        basepath += f'{self.sampling_file}_{self.start_date}_{self.end_date}'
        save_patch_arrays(data, basepath, self.label_class)

        fig = plot_numpy_grid(np.clip(data[:,:,:,(3,2,1)] / 2500, 0, 1))
        fig.savefig(f'{basepath}.png', bbox_inches='tight', pad_inches=0)
        plt.show()

        self.data = data


def plot_numpy_grid(patches):
    num_img = int(np.ceil(np.sqrt(len(patches))))
    padding = 1
    h,w,c = patches[0].shape
    mosaic = np.zeros((num_img * (h + padding), num_img * (w + padding), c))
    counter = 0
    for i in range(num_img):
        for j in range(num_img):
            if counter < len(patches):
                mosaic[i * (h + padding):(i + 1) * h + i * padding, 
                    j * (w + padding):(j + 1) * w + j * padding] = patches[counter]
            else:
                mosaic[i * (h + padding):(i + 1) * h + i * padding, 
                    j * (w + padding):(j + 1) * w + j * padding] = np.zeros((h,w,c))    
            counter += 1
            
    fig, ax = plt.subplots(figsize=(num_img, num_img), dpi=150)
    ax.axis('off')
    ax.imshow(mosaic)
    return fig

def save_patch_arrays(data, basepath, label_class):
    with open(basepath + '_patch_arrays.pkl', "wb") as f:
        pickle.dump(data, f)
    with open(basepath + '_patch_array_labels.pkl', "wb") as f:
        pickle.dump([label_class] * len(data), f)