import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn.manifold import TSNE
from tqdm import tqdm
from scripts import dl_utils

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

def create_img_stack(patch_history):
    img_stack = []
    for date in patch_history:
        for site in patch_history[date]:
            spectral_stack = []
            band_shapes = [np.shape(patch_history[date][site][band])[0] for band in band_descriptions]
            if np.array(band_shapes).all() > 0:
                for band in band_descriptions:
                    spectral_stack.append(patch_history[date][site][band])
                cloud_percentage = 1 - np.sum(np.array(spectral_stack) > 0) / np.size(spectral_stack)
                if cloud_percentage < 0.2:
                    img_stack.append(np.rollaxis(np.array(spectral_stack), 0, 3))

    min_x = np.min([np.shape(img)[0] for img in img_stack])
    min_y = np.min([np.shape(img)[1] for img in img_stack])
    img_stack = [img[:min_x, :min_y, :] for img in img_stack]
    return img_stack

def create_img_stack_mean(patch_history, cloud_threshold=0.2):
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
        min_dim = np.min(np.shape(img_stack)[1:3])
        img_stack = np.array(img_stack)[:, :min_dim, :min_dim, :]

        masked_img = []
        for img in img_stack:
            masked_img.append(np.ma.masked_where(img < 0, img))

        masked_mean = np.ma.mean(masked_img, axis=0)

        num_cloudy_pixels = np.sum(masked_mean.mask)
        cloud_fraction = num_cloudy_pixels / np.size(masked_mean)
        if cloud_fraction < cloud_threshold:
            mean_stack.append(masked_mean.data)
    return np.array(mean_stack)

def plot_image_grid(patches, labels=False, file_path=None):
    num_img = int(np.ceil(np.sqrt(len(patches))))
    plt.figure(figsize=(num_img, num_img), dpi=100)
    for index, img in enumerate(tqdm(patches)):
        plt.subplot(num_img, num_img, index + 1)
        if np.ma.is_masked(img):
            img[img.mask] = 0
        plt.imshow(np.clip(normalize(img[:,:,3:0:-1]), 0, 1))
        if len(np.shape(labels)) > 0:
            plt.title(labels[index])
        plt.axis('off')
    plt.tight_layout()
    if file_path:
        title = os.path.basename(file_path)
        plt.suptitle(title, size = num_img * 12 / 7, y=1.02)
        plt.savefig(file_path + '.png', bbox_inches='tight')
    plt.show()

def visualize_history(patch_history, file_path=None):
    img_stack = create_img_stack_mean(patch_history)
    rgb_img = create_rgb(img_stack)
    plot_image_grid(rgb_img, file_path=file_path)

def normalize(x):
    return (np.array(x)) / (3000)

def animate_patch_history(data, file_path, max_cloud=1):
    """
    Used for visualization and debugging. Takes a history dictionary and outputs a video
    for each timestep at each site in the history.
    """
    fig, ax = plt.subplots(dpi=100, facecolor=(1,1,1))
    ax.set_axis_off()
    images = []
    init_date = list(data.keys())[0]
    for site_name in data[init_date]:
        for date in data.keys():
            ax.set_title(os.path.basename(file_path)[:-4])
            hyperpatch = data[date][site_name]
            rgb = np.stack((hyperpatch['B4'], hyperpatch['B3'], hyperpatch['B2']), axis=-1)
            if len(rgb) > 0:
                if np.sum(rgb <= 0) / np.size(rgb) <= max_cloud:
                    rgb_stretch = stretch_histogram(normalize(rgb), 0.1, 1.0, gamma=1.2)
                    im = plt.imshow(rgb_stretch, animated=True)
                    images.append([im])
    fig.tight_layout()
    ani = animation.ArtistAnimation(fig, images, interval=100, blit=True, repeat_delay=500)
    ani.save(file_path)

def animate_patch(data, file_path, cloud_threshold=0.1, stretch=True, interval=100, size=4):
    """
    Used for visualization and debugging. Takes a history dictionary and outputs a video
    for each timestep at each site in the history.
    """
    fig, ax = plt.subplots(dpi=100, facecolor=(1,1,1), figsize=(size, size))
    ax.set_axis_off()
    images = []
    ax.set_title(os.path.basename(file_path)[:-4])
    for img in data:
        if np.sum(img.mask) / np.size(img.mask) < cloud_threshold:
            rgb = normalize(img[:,:,3:0:-1])
            if stretch:
                rgb = stretch_histogram(rgb, 0.1, 1.0, gamma=1.2)
            im = plt.imshow(np.clip(rgb, 0, 1), animated=True)
            images.append([im])
    fig.tight_layout()
    ani = animation.ArtistAnimation(fig, images, interval=interval, blit=True, repeat_delay=500)
    ani.save(file_path)

def compare_networks(pairs, models, names=None, threshold=0.8, plot=True):
    """
    Compare predictions on a spectrogram pair patch for a list of networks
    """
    rgb = normalize(np.ma.mean(pairs[:][0], axis=0))[:,:,3:0:-1]
    overlays = []
    preds = []
    for model in models:
        pred_stack = [dl_utils.predict_spectrogram(pair, model) for pair in pairs]
        pred = np.ma.mean(pred_stack, axis=0)
        preds.append(pred)
        overlay = np.copy(rgb)
        overlay[pred > threshold, 0] = .9
        overlay[pred > threshold, 1] = 0
        overlay[pred > threshold, 2] = .1
        overlays.append(overlay)

    if plot:
        num_plots = len(models) + 1
        plt.figure(figsize=(num_plots * 5, 10), dpi=100)

        plt.subplot(2, num_plots, 1)
        plt.title('RGB Mean')
        plt.imshow(np.clip(rgb, 0, 1))
        plt.axis('off')

        for num in range(len(models)):
            plt.subplot(2, num_plots, num + 2)
            plt.imshow(preds[num], vmin=0, vmax=1, cmap='RdBu_r')
            plt.axis('off')
            if names:
                plt.title(f'{names[num]} Preds')
        for num in range(len(models)):
            plt.subplot(2, num_plots,  num_plots + num + 2)
            plt.imshow(np.clip(overlays[num], 0, 1))
            plt.axis('off')
            if names:
                plt.title(f'{names[num]} Thresh {threshold}')
            else:
                plt.title(f'Thresh {threshold}')
    plt.tight_layout()
    plt.show()

    return rgb, preds, overlays
