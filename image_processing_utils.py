
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import PIL
import skimage
from skimage.transform import resize
from skimage.metrics import mean_squared_error
from PIL import Image
# from simulator import *
import pandas as pd

def load_image(path):
    # Load the image
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    # Check the number of channels
    if len(image.shape) == 2:
        target_image = image.astype(float) / 255

    elif len(image.shape) == 3:
        print("The target image has 3 channels, converting to grayscale by default")
        target_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(float) / 255

    else:
        raise ValueError("The target image should have 1 channel, please check its bit depth")

    return target_image


def resize_image(image, x, y):
    resized_image = skimage.transform.resize(image, output_shape = (x, y), mode='constant', anti_aliasing=True)
    return resized_image



def center_square_crop(img: np.ndarray) -> np.ndarray:
    """
    Return the largest centred square that fits inside `img`.

    Parameters
    ----------
    img : np.ndarray
        Image array of shape (H, W) or (H, W, C).

    Returns
    -------
    cropped : np.ndarray
        Square crop of side length  S = min(H, W), centred in `img`.
    """
    H, W = img.shape[:2]
    S = min(H, W)                     # largest square side

    top  = (H - S) // 2
    left = (W - S) // 2

    return img[top:top+S, left:left+S, ...]   # works for 2-D or 3-D arrays


class TargetImage():

    def __init__(
            self,
            image_array,
            ):

        self.image_array = image_array

    def show_image(self):
        plt.imshow(self.image_array)
        plt.axis('off')  # Turn off axis
        plt.show()

