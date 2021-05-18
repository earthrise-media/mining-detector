
import json
import os
import sys
import pickle

from geojson import Feature, FeatureCollection, dump
import geopandas
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import box, shape, Point, Polygon
from sklearn.manifold import TSNE

from scripts.get_s2_data_ee import get_history, get_history_polygon, get_pixel_vectors

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

def sample_polygon(polygon_coords, num_samples, rect_width = 0.0075, min_intersection=0.25, plot=False):
    """
    Given a polygon, generate a list of coordinate centers of patches that intersect with the patch
    by a given value
    """
    site_polygon = Polygon(polygon_coords)
    min_lon, min_lat, max_lon, max_lat = site_polygon.bounds
    valid_points = []

    polygon_area = site_polygon.area
    while len(valid_points) < num_samples:
        rand_point = Point(np.random.uniform(min_lon - rect_width / 4, max_lon + rect_width / 4),
                           np.random.uniform(min_lat - rect_width / 4, max_lat + rect_width / 4))
        rect = box(rand_point.x - rect_width / 2, rand_point.y - rect_width / 2, rand_point.x + rect_width / 2, rand_point.y + rect_width / 2)
        rect_area = rect.area

        if rect_area > polygon_area:
            coverage_area = polygon_area
        else:
            coverage_area = rect.area

        if site_polygon.intersection(rect).area / coverage_area > 0.25:
            valid_points.append(rand_point)

        if plot:
            if site_polygon.intersection(rect).area / coverage_area > 0.25:
                plt.plot(*rect.exterior.xy, c='C0', alpha=0.5)
                plt.scatter(rand_point.x, rand_point.y, c='C0')
            else:
                plt.plot(*rect.exterior.xy, c='r', alpha=0.25)
                plt.scatter(rand_point.x, rand_point.y, c='r')
                #plt.title(f"Polygon Overlap: {site_polygon.intersection(rect).area / site_polygon.area:.0%}")
    if plot:
        plt.plot(*site_polygon.exterior.xy, c='k', linewidth=2)
        plt.axis('equal')
        plt.show()
    sample_points = [[point.x, point.y] for point in valid_points]
    return sample_points

def sample_geojson(geojson, num_samples, rect_width):
    """Generate list of sampling coordinates from geojson polygons"""
    coords = []
    names = []
    for index, site in enumerate(geojson['features']):
        coords += sample_polygon(site['geometry']['coordinates'][0], num_samples, rect_width)
        names += [f"{index}_{site_num}" for site_num in range(num_samples)]
    print(len(coords), "sampling sites selected from polygons")
    return coords, names


def get_image_stack(coords, start_date='2020-05-01', rect_width=0.025, scale=10, num_months=3, cloud_mask=True):
    """
    Download data from GEE. Return a detailed patch history array, and also a stack of
    images with dimensions (n,width,height,bands)
    """
    names = ['sample_' + str(i) for i in range(len(coords))]
    history = get_history(coords,
                          names,
                          rect_width,
                          start_date=start_date,
                          num_months=num_months,
                          #scale=rect_width * (100 / 0.025)
                          scale=scale,
                          cloud_mask=cloud_mask
                         )
    img_stack = create_img_stack_mean(history)
    print("Image shape before cropping:", img_stack[0].shape)
    min_dim = np.min([img.shape[:2] for img in img_stack])
    img_stack = [img[:min_dim, :min_dim, :] for img in img_stack]

    return history, img_stack

def create_img_stack_mean(patch_history):
    """
    Process a dictionary of patches into single images with cloudiness below a threshold and
    averaged across all time periods in the dataset
    """
    mean_stack = []
    dates = list(patch_history.keys())
    for site in patch_history[dates[0]]:
        img_stack = []
        for date in dates:
            spectral_stack = []
            band_shapes = [np.shape(patch_history[date][site][band])[0] for band in band_descriptions]
            if np.array(band_shapes).all() > 0:
                for band in band_descriptions:
                    spectral_stack.append(patch_history[date][site][band])
                img_stack.append(np.rollaxis(np.array(spectral_stack), 0, 3))

        masked_img = []
        for img in img_stack:
            masked_img.append(np.ma.masked_where(img < 0, img))

        masked_mean = np.ma.mean(masked_img, axis=0)

        num_cloudy_pixels = np.sum(masked_mean.mask)
        cloud_fraction = num_cloudy_pixels / np.size(masked_mean)

        print("Cloud Fraction", cloud_fraction)
        if cloud_fraction < 0.2:
            mean_stack.append(masked_mean.data)

    return np.array(mean_stack)


def stretch_histogram(array, min_val=0.1, max_val=0.75, gamma=1.2):
    clipped = np.clip(array, min_val, max_val)
    stretched = np.clip((clipped - min_val) / (max_val - min_val) ** gamma, 0, 1)
    return stretched

def create_rgb(img_array):
    """
    Create three-channel RGB images for visualization from an image stack
    """
    rgb_img = []
    for img in img_array:
        rgb = np.stack((img[:,:,3],
                        img[:,:,2],
                        img[:,:,1]), axis=-1)
        rgb = stretch_histogram(normalize(rgb), 0.1, 1.0, gamma=1.2)
        rgb_img.append(rgb)
    return rgb_img

def create_img_vectors(img_array):
    img_vecs = []
    for img in img_array:
        img_vecs.append(img.flatten())
    return np.array(img_vecs)

def plot_similar_images(img_stack, title, save=True):
    reducer = TSNE(n_components=1)
    reduced = reducer.fit_transform(normalize(create_img_vectors(img_stack)))
    input_img = create_rgb(img_stack)
    num_img = int(np.ceil(np.sqrt(len(input_img))))

    plt.figure(figsize=(num_img, num_img), dpi=100)
    for img_index, sort_index in enumerate(reduced[:,0].argsort()):
        plt.subplot(num_img, num_img, img_index + 1)
        plt.imshow(input_img[sort_index])
        plt.axis('off')
    plt.tight_layout()
    plt.suptitle(title, size = num_img * 12 / 7, y=1.02)
    if save:
        plt.savefig('../figures/' + title + ' Similarity.png', bbox_inches='tight')
    plt.show()

def plot_image_grid(rgb_img, title, save=True):
    num_img = int(np.ceil(np.sqrt(len(rgb_img))))
    plt.figure(figsize=(num_img,num_img), dpi=100)
    for index, img in enumerate(rgb_img):
        plt.subplot(num_img, num_img, index + 1)
        plt.imshow(img)
        plt.axis('off')
    plt.tight_layout()
    plt.suptitle(title, size = num_img * 12 / 7, y=1.02)
    if save:
        plt.savefig('../figures/' + title + ' Grid.png', bbox_inches='tight')
    plt.show()

def normalize(x):
    return (np.array(x)) / (3000)
