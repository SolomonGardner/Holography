import matplotlib.pyplot as plt
import pandas as pd
import algorithms
import loss_functions
import genetic_algorithm_utils
import numpy as np
from PIL import Image
import matplotlib.colors as mcolors
import eval_metrics
import image_processing_utils
from skimage.metrics import structural_similarity, peak_signal_noise_ratio, mean_squared_error
from pathlib import Path
import fastparquet
import numpy as np
import cv2
from scipy.ndimage import zoom     # or cv2.resize, PIL etc.


path = 'diverse_images/0009.png'
path=fp
loaded_image = image_processing_utils.load_image(path)
loaded_image = image_processing_utils.center_square_crop(loaded_image)
target_image = image_processing_utils.TargetImage(image_array=loaded_image)

HGH = algorithms.HologramGenerationAlgo(image_array=target_image)

R2, loss2, H = HGH.gradient_descent_cont(n_iter=50, loss_function=loss_functions.mse_loss_and_grad_new, roi=(slice(None), slice(None)))

R2 = np.clip(R2,   0.0, 1.0).astype(np.float32)

structural_similarity_index = structural_similarity(
        target_image.image_array,
        R2,
        data_range=1.0,          # because inputs are in [0,1]
        multichannel=False
    )
#scale → 0…255  and convert to 8-bit unsigned integers
H_uint8 = (H * 255.0).round().astype(np.uint8)

# wrap as a grayscale ("L") image and save losslessly
img = Image.fromarray(H_uint8, mode="L")   # ‘L’ = 8-bit, 1 channel
img.save("hologram.png", format="PNG", compress_level=0)  # no zlib loss


