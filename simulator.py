import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity

import algorithms
import loss_functions
import image_processing_utils

# Config
INPUT_PATH = 'diverse_images/0009.png'
OUTPUT_PATH = 'hologram.png'
ITERATIONS = 50

def run_hologram_generation(input_path: str, output_path: str, n_iter: int):
    # Load and preprocess
    img = image_processing_utils.load_image(input_path)
    img = image_processing_utils.center_square_crop(img)
    target = image_processing_utils.TargetImage(image_array=img)

    # Generate hologram
    algo = algorithms.HologramGenerationAlgo(image_array=target)
    R2, loss_values, H = algo.gradient_descent_cont(
        n_iter=n_iter,
        loss_function=loss_functions.mse_loss_and_grad_new,
        roi=(slice(None), slice(None)) # change to set region of interest in img
    )

    # Clip result to [0,1]
    R2 = np.clip(R2, 0.0, 1.0).astype(np.float32)

    # Compute SSIM
    ssim = structural_similarity(
        target.image_array,
        R2,
        data_range=1.0,
        multichannel=False
    )
    print(f"Structural Similarity Index (SSIM): {ssim:.4f}")

    # Save hologram image
    H_uint8 = (H * 255.0).round().astype(np.uint8)
    hologram_img = Image.fromarray(H_uint8, mode="L")
    hologram_img.save(output_path, format="PNG", compress_level=0)
    print(f"Hologram saved to: {output_path}")


def main():
    run_hologram_generation(INPUT_PATH, OUTPUT_PATH, ITERATIONS)


if __name__ == "__main__":
    main()


