import numpy as np
import pywt
from skimage.metrics import structural_similarity, peak_signal_noise_ratio, mean_squared_error


def mse(image_amplitude, target_image_amplitude):
    loss = mean_squared_error(image_amplitude, target_image_amplitude) / np.sum(target_image_amplitude**2) # do we need the denominator?
    return loss


def ssim_loss(image_amplitude, target_image_amplitude):
    """
    1 − SSIM  → 0 when the two images are identical.
    """
    ssim_val = structural_similarity(
        target_image_amplitude,
        image_amplitude,
        data_range=1.0,          # because inputs are in [0,1]
        multichannel=False
    )
    return 1.0 - ssim_val       # convert similarity → residual


# PSNR “loss”  (smaller = better, 0 is perfect) 
def psnr_loss(image_amplitude, target_image_amplitude):
    """
    −PSNR (dB)  → 0 dB when identical, increasingly negative as images differ.
    """
    psnr_val = peak_signal_noise_ratio(
        target_image_amplitude,
        image_amplitude,
        data_range=1.0
    )
    return -psnr_val            # higher PSNR is better → negate for a loss


def wncc(image1, image2, wavelet='db2', level=None):
    """
    Calculate the Normalized Cross-Correlation (NCC) between two images in the wavelet domain.

    Parameters:
    - image1, image2: Input images (numpy arrays) to compare.
    - wavelet: Type of wavelet to use (default is 'db2' - Daubechies 2).
    - level: Decomposition level (default is maximum level based on data).

    Returns:
    - transformed_score: The NCC score in the wavelet domain.
    """
    if image1.shape != image2.shape:
        raise ValueError("Images must have the same dimensions for NCC calculation.")

    # Compute the 2D Discrete Wavelet Transform of both images
    coeffs1 = pywt.wavedec2(image1, wavelet=wavelet, level=level)
    coeffs2 = pywt.wavedec2(image2, wavelet=wavelet, level=level)

    # Flatten the wavelet coefficients into 1D arrays
    coeffs1_array = []
    coeffs2_array = []

    # coeffs[0] is the approximation coefficients, coeffs[1:] are detail coefficients
    coeffs1_array.extend(coeffs1[0].flatten())
    coeffs2_array.extend(coeffs2[0].flatten())

    for detail_level in range(1, len(coeffs1)):
        cH1, cV1, cD1 = coeffs1[detail_level]
        cH2, cV2, cD2 = coeffs2[detail_level]

        coeffs1_array.extend(cH1.flatten())
        coeffs1_array.extend(cV1.flatten())
        coeffs1_array.extend(cD1.flatten())

        coeffs2_array.extend(cH2.flatten())
        coeffs2_array.extend(cV2.flatten())
        coeffs2_array.extend(cD2.flatten())

    coeffs1_array = np.array(coeffs1_array)
    coeffs2_array = np.array(coeffs2_array)

    # Calculate the NCC between the wavelet coefficients
    mean1 = np.mean(coeffs1_array)
    mean2 = np.mean(coeffs2_array)
    coeffs1_centered = coeffs1_array - mean1
    coeffs2_centered = coeffs2_array - mean2
    numerator = np.sum(coeffs1_centered * coeffs2_centered)
    denominator = np.sqrt(np.sum(coeffs1_centered ** 2) * np.sum(coeffs2_centered ** 2))

    if denominator == 0:
        return 0  # Avoid division by zero; implies no variation in one or both images
    else:
        ncc_score = numerator / denominator
        # Transform the score so that closer to 0 is better
        transformed_score = 1 - abs(ncc_score)
        return transformed_score


def mse_loss_grad(A: np.ndarray, T: np.ndarray):
    diff = A - T
    loss = float(np.mean(diff**2))
    g_A  = (2.0 / diff.size) * diff     # analytic ∂MSE/∂A

    return loss, g_A.astype(np.float32)


