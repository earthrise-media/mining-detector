import pickle

from descarteslabs.geo import DLTile
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np

import gee_data

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
        # create DLTiles for each sampling location
        tiles = [DLTile.from_latlon(lat, lon, resolution=10, tilesize=self.patch_size+2, pad=0) for lat, lon in zip(lats, lons)]
        print(f"{len(tiles)} tiles to download")
        self.tiles = tiles

    def get_patches(self):
        # get data for each tile
        
        self.create_tiles()

        print(f'Getting data from {self.start_date} to {self.end_date}')
        data = gee_data.get_image_data(self.tiles, self.start_date, self.end_date, self.clear_threshold)
        data = np.array([gee_data.pad_patch(patch, self.patch_size) for patch in data])
        print(f'Retrieved {data.shape[0]} patches')

        fig = plot_numpy_grid(np.clip(data[:,:,:,(3,2,1)] / 2500, 0, 1))
        fig.savefig(f'../data/training_data/{self.sampling_file}_{self.start_date}_{self.end_date}.png', bbox_inches='tight', pad_inches=0)
        plt.show()

        # save the data
        basepath = f'../data/training_data/{self.sampling_file}_{self.start_date}_{self.end_date}'
        save_patch_arrays(data, basepath, self.label_class)
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