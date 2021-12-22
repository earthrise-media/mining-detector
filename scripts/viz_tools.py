import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

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

def normalize(x):
    return (np.array(x)) / (3000)

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