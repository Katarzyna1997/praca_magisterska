import numpy as np
import matplotlib.pyplot as plt

from PIL import Image


def load_image(path):
    return np.array(Image.open(path))


def plot_sample(lr, sr, sr2):
    plt.figure(figsize=(20, 10))

    images = [lr, sr, sr2]
    titles = ['Low-Resolution', f'Super-Resolution(ESPCN)', f'Super-Resolution(RDN)']

    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(1, 2, i+1)
        plt.imshow(img)
        plt.title(title)
        plt.xticks([])
        plt.yticks([])
