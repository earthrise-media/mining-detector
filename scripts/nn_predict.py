import argparse
import os

import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tqdm import tqdm

from get_s2_data_ee import get_history, get_pixel_vectors

def normalize(x):
    return (np.array(x) - 0) / (3000 - 0)

def stretch_histogram(array, min_val=0.1, max_val=0.75, gamma=1.2):
    clipped = np.clip(array, min_val, max_val)
    stretched = (clipped - min_val) / (max_val - min_val) ** gamma
    return stretched

def make_predictions(model_path, data, site_name, threshold):
    test_image = data
    model = keras.models.load_model(model_path)

    rgb_stack = []
    preds_stack = []
    threshold_stack = []
    print("Making Predictions")
    for month in tqdm(list(test_image.keys())):
        test_pixel_vectors, width, height = get_pixel_vectors(test_image, month)
        if width > 0:
            test_pixel_vectors = normalize(test_pixel_vectors)

            r = np.reshape(np.array(test_pixel_vectors)[:,3], (width, height))
            g = np.reshape(np.array(test_pixel_vectors)[:,2], (width, height))
            b = np.reshape(np.array(test_pixel_vectors)[:,1], (width, height))
            rgb = np.moveaxis(np.stack((r,g,b)), 0, -1)
            rgb_stack.append(rgb)

            preds = model.predict(np.expand_dims(test_pixel_vectors, axis=-1))
            preds_img = np.reshape(preds, (width, height, 2))[:,:,1]
            preds_stack.append(preds_img)

            thresh_img = preds_img >= threshold
            threshold_stack.append(thresh_img)

    output_dir = '../notebooks/figures/neural_network'
    if not os.path.exists(output_dir):
            os.mkdir(output_dir)


    rgb_median = np.median(rgb_stack, axis=0)
    preds_median = np.median(preds_stack, axis=0)
    threshold_median = np.median(threshold_stack, axis=0)

    plt.figure(dpi=150, figsize=(15,5))

    plt.subplot(1,3,1)
    plt.imshow(stretch_histogram(rgb_median))
    plt.title(f'{site_name} Median', size=8)
    plt.axis('off')

    plt.subplot(1,3,2)
    plt.imshow(preds_median, vmin=0, vmax=1, cmap='seismic')
    plt.title('Classification Median', size=8)
    plt.axis('off')

    plt.subplot(1,3,3)
    plt.imshow(threshold_median, vmin=0, vmax=1, cmap='gray')
    plt.title(f"Positive Pixels Median: Threshold {threshold}", size=8)
    plt.axis('off')

    title = f"{site_name} - Median Values - Neural Network Classification - Threshold {threshold}"
    plt.suptitle(title, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, title + '.png'), bbox_inches='tight')
    plt.close()


    fig, ax = plt.subplots(dpi=200, facecolor=(1,1,1), figsize=(4,4))
    ax.set_axis_off()
    clipped_img = np.moveaxis([channel * (preds_median >= 0) for channel in np.moveaxis(rgb_median, -1, 0)], 0, -1)
    img = plt.imshow(np.clip(stretch_histogram(clipped_img), 0, 1))
    ax.set_title('Threshold 0.00', size=10)
    plt.tight_layout()

    def animate(i):
        i /= 100
        clipped_img = np.moveaxis([channel * (preds_median >= i) for channel in np.moveaxis(rgb_median, -1, 0)], 0, -1)
        img.set_data(np.clip(stretch_histogram(clipped_img), 0, 1))
        #img.set_data((preds_stack > i) * 1)
        ax.set_title(f"{site_name} Threshold {i:.2f}", size=10)
        return img,

    ani = animation.FuncAnimation(fig, animate, frames=100, interval=60, blit=True, repeat_delay=500)
    ani.save(os.path.join(output_dir, site_name + '_threshold_visualization' + '.mp4'))
    plt.close()

    return rgb_median, preds_median, threshold_median

def main():
    parser = argparse.ArgumentParser(description='Configure patch prediction')
    parser.add_argument('--coords', nargs='+', required=True, type=float, help='Lat Lon of patch center')
    parser.add_argument('--width', type=float, required=False, default=0.002, help='Width of patch in degrees. Max 0.03')
    parser.add_argument('--network', type=str, required=True, help='Path to neural network')
    parser.add_argument('--threshold', type=float, required=False, default=0.95, help='Classifier masking threshold')
    args = parser.parse_args()
    if args.width and args.width > 0.03:
        parser.error("Maximum patch width is 0.03")

    coords = args.coords
    print(coords)
    lat = coords[0]
    lon = coords[1]
    width = args.width
    model_path = args.network
    threshold = args.threshold

    name = f"{lat:.2f}, {lon:.2f}, {width} patch"

    patch_history = get_history([[lon, lat]], [name], width)
    rgb_median, preds_median, threshold_median = make_predictions(model_path, patch_history, name, threshold)

if __name__ == '__main__':
    main()
