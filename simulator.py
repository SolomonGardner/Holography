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

ssim_mse = []                       # will hold one dict per image
ssim_wncc = []                       # will hold one dict per image

root = Path("diverse_images")   # folder that holds 0009.png … 0800.png

# range(9, 801)  →  9 … 800 inclusive
for idx in range(6, 7): #801
    fname = f"{idx:04d}.png"   # zero-pad to four digits
    fp = root / fname
    if not fp.exists():        # guard against missing files
        print(f"⚠️  {fp} not found; skipping")
        continue
    print(fp)

    #path = 'diverse_images/0009.png'
    path=fp
    loaded_image = image_processing_utils.load_image(path)
    print(loaded_image.shape)
    #loaded_image = image_processing_utils.resize_image(loaded_image, x=1080, y=1920)
    loaded_image = image_processing_utils.center_square_crop(loaded_image)

    loaded_image = np.zeros((100,100))
    loaded_image[35:65, 35:65] = 1
    print(loaded_image)

    # plt.close('all')
    # plt.imshow(loaded_image, cmap='gray')
    # plt.savefig('tiger_target_image.png')
    #plt.show()
    target_image = image_processing_utils.TargetImage(image_array=loaded_image)

    HGH = algorithms.HologramGenerationAlgo(image_array=target_image)

    H_list = []

    R1_list = []
    R2_list = []
    for i in range(1):
        print(i)

        R2, loss2, H_cwncc = HGH.gradient_descent_cont(n_iter=50, loss_function=loss_functions.cwncc_loss_and_grad_new,
                                              roi=(slice(None), slice(None)))
        # R1_list.append(R1)
        R2_list.append(R2)
        H_list.append(H_cwncc)

    # stack1 = np.stack(R1_list, axis=0)
    # R1 = stack1.mean(axis=0)
    stack2 = np.stack(R2_list, axis=0)
    R2 = stack2.mean(axis=0)

    plt.close('all')
    plt.imshow(R2, cmap='gray')
    plt.show()

    # img_R2 = Image.fromarray(R2, mode="L")   # ‘L’ = 8-bit, 1 channel
    # img_R2.save("recon_square_03062025_redpanda.png", format="PNG", compress_level=0)  # no zlib loss


    ### clipping
    # R1 = np.clip(R1, 0.0, 1.0).astype(np.float32)
    R2 = np.clip(R2,   0.0, 1.0).astype(np.float32)

    ssim_val_2 = structural_similarity(
            target_image.image_array,
            R2,
            data_range=1.0,          # because inputs are in [0,1]
            multichannel=False
        )

    # print(ssim_val_1)
    print(ssim_val_2)

    # ssim_mse.append(ssim_val_1)
    ssim_wncc.append(ssim_val_2)


data = pd.DataFrame()
# data['mse ssim score'] = ssim_mse
data['wncc ssim score'] = ssim_wncc

# data.to_parquet('relative_ssim_scores_full.parquet')

# print(data['mse ssim score'].mean())
print(data['wncc ssim score'].mean())

# convert to greyscale
# imshow, save to png

H_clipped = H_list[0]


# 2.  scale → 0…255  and convert to 8-bit unsigned integers
H_uint8 = (H_clipped * 255.0).round().astype(np.uint8)

# 3.  wrap as a grayscale ("L") image and save losslessly
# img = Image.fromarray(H_uint8, mode="L")   # ‘L’ = 8-bit, 1 channel
# img.save("hologram_square_03062025_bears.png", format="PNG", compress_level=0)  # no zlib loss


