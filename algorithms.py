import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import PIL
import skimage
from skimage.transform import resize
from PIL import Image
from tqdm import tqdm
import math
import image_processing_utils
import loss_functions
import genetic_algorithm_utils
import torch
from tqdm import tqdm


def pad_to_square_complex(arr: np.ndarray):
    """
    Zero-pad a 2-D complex array so it becomes square.

    Returns
    -------
    arr_pad : np.ndarray
        Padded array (S × S).
    offsets : tuple[int, int]
        (pad_top, pad_left) — how many rows/cols were added above/left
        of the original array.
    """
    H, W = arr.shape
    S = max(H, W)
    pad_top = (S - H) // 2
    pad_bottom = S - H - pad_top
    pad_left = (S - W) // 2
    pad_right = S - W - pad_left

    arr_pad = np.pad(arr,
                     pad_width=((pad_top, pad_bottom),
                                (pad_left, pad_right)),
                     mode="constant",
                     constant_values=0)

    return arr_pad, (pad_top, pad_left)

class HologramGenerationAlgo():
    """
    The HologramGenerationAlgo class is designed to take a TargetImage object which contains an image
    array. We can then apply a hologram generation algorithm of our choice, using a loss function, and
    other params of our choice.

    """

    def __init__(
            self,
            image_array: image_processing_utils.TargetImage,
            verbose: bool = True,
    ):
        self.image_array = image_array
        self.verbose = verbose

    def direct_binary_search(self, loss_function, n_iter=2000, roi=(slice(None), slice(None))):

        image = self.image_array.image_array
        TargetAmplitude = np.sqrt(image)
        H = np.round(np.random.rand(*TargetAmplitude.shape))
        R = abs(np.fft.fftshift(np.fft.fft2(np.fft.fftshift(np.exp(1j * H * np.pi)))))
        R = abs(R) * np.sqrt(np.sum(np.sum(TargetAmplitude ** 2)) / np.sum(np.sum(np.abs(R) ** 2)))
        L = loss_function(R[roi], TargetAmplitude[roi])

        loss_list = []  # Record loss (for plotting)
        for n in tqdm(range(n_iter), disable=not self.verbose, desc="Processing"):
            random_location = np.random.randint(0, H.shape[0]), np.random.randint(0, H.shape[1])
            H_n = H.copy()
            H_n[random_location] = 1 - H[random_location]

            # Calculate the loss function for the new hologram
            R_n = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(np.exp(1j * H_n * np.pi))))
            R_n = np.abs(R_n) * np.sqrt(
                np.sum(np.sum(TargetAmplitude ** 2)) / np.sum(np.sum(np.abs(R_n) ** 2)))  # conservation of energy
            L_n = loss_function(R_n[roi], TargetAmplitude[roi])

            if L_n < L:
                H = H_n
                R = R_n
                L = L_n

            loss_list.append(L_n)

        return R, loss_list

    def simulated_annealing(self, loss_function, n_iter=2000, roi=(slice(None), slice(None))):

        image = self.image_array
        TargetAmplitude = np.sqrt(image.image_array)

        H = np.round(np.random.rand(*TargetAmplitude.shape))

        R = abs(np.fft.fftshift(np.fft.fft2(np.fft.fftshift(np.exp(1j * H * np.pi)))))
        R = abs(R) * np.sqrt(np.sum(np.sum(TargetAmplitude ** 2)) / np.sum(np.sum(np.abs(R) ** 2)))

        # Calculate initial loss
        L = loss_function(TargetAmplitude[roi], R[roi])

        initial_SA_thresh = 0.1
        exponent = math.log(1 / initial_SA_thresh) / (n_iter - 1)
        SA_thresh = initial_SA_thresh

        loss_list = []

        for n in tqdm(range(n_iter), disable=not self.verbose, desc="Processing"):

            SA_thresh = SA_thresh * math.exp(exponent)

            # Flip random pixel
            random_location = np.random.randint(0, H.shape[0]), np.random.randint(0, H.shape[1])
            H_n = H.copy()
            H_n[random_location] = 1 - H[random_location]

            # Calculate the loss function for the new hologram
            R_n = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(np.exp(1j * H_n * np.pi))))
            R_n = np.abs(R_n) * np.sqrt(
                np.sum(np.sum(TargetAmplitude ** 2)) / np.sum(np.sum(np.abs(R_n) ** 2)))  # conservation of energy
            L_n = loss_function(TargetAmplitude[roi], R_n[roi])

            if L_n < L:
                H = H_n
                R = R_n
                L = L_n
            else:
                p_n = np.random.rand()
                print(SA_thresh)
                if p_n > SA_thresh:
                    H = H_n
                    R = R_n
                    L = L_n

            loss_list.append(L_n)

        return R, loss_list

    def genetic_algorithm(self, crossover_function, parent_selection_function, N=5000, N_iter=1000,
                          roi=(slice(None), slice(None))):

        T = np.sqrt(self.image_array.image_array)
        H_list = [(np.random.rand(*T.shape)) for _ in range(N)] # continuous phase
        error_list = []
        R_list = []
        R_best = None
        H_best = None
        lowest_mse = float('inf')  # Set initial lowest MSE to infinity for comparison

        # Genetic Algorithm Loop
        for iteration in tqdm(range(N_iter), disable=not self.verbose, desc="Processing"):
            mse_list = []
            for H in H_list:

                R_n = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(np.exp(1j * H * np.pi))))
                R_n = np.abs(R_n) * np.sqrt(np.sum(np.sum(T ** 2)) / np.sum(np.sum(np.abs(R_n) ** 2)))

                if iteration > 497:
                    R_list.append(R_n)
                mse = loss_functions.mean_squared_error(T[roi], R_n[roi])# / np.sum(T[roi] ** 2)  # FIX!!!
                mse_list.append(mse)

                if mse < lowest_mse:
                    lowest_mse = mse
                    R_best = R_n
                    H_best = H

            n_splits = 8 - (iteration // 50)
            if n_splits < 4:
                n_splits = 4

            error_list.append(np.mean(mse_list))
            H_parents = parent_selection_function(mse_list, H_list)
            H_children = crossover_function(H_parents, n_splits=n_splits)

            H_list = H_children[:N]

            if len(error_list) >= 4 and len(set(error_list[-4:])) == 1:
                print("Stopping loop, MSE values have been the same for 4 consecutive iterations.")
                break

        R_list = []
        for H in H_list:
            R_n = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(np.exp(1j * H * np.pi))))
            R_n = np.abs(R_n) * np.sqrt(np.sum(np.sum(T ** 2)) / np.sum(np.sum(np.abs(R_n) ** 2)))

            R_list.append(R_n)

        return R_best, error_list, H_best, H_list, R_list


    def one_step_phase_retrieval(self, N=50):

        TargetImage = np.sqrt(self.image_array.image_array)
        TargetAmplitude = np.sqrt(TargetImage)  # target amplitude is the square root of intensity
        T = TargetAmplitude

        R_total = np.zeros_like(T)  # Create an array of zeros with the same size as T
        FreemanHolo = np.zeros((T.shape[0], T.shape[1], 3), dtype=np.uint8)

        for HoloChannelIndex in range(1, 4):
            HoloPerChannel = 0
            for PerChannelFrameIndex in range(1, N + 1):
                holo_frame_i = (HoloChannelIndex - 1) * 8 + PerChannelFrameIndex  # Compute current subframe index

                random_phase = np.exp(1j * 2 * np.pi * np.random.rand(*T.shape))
                E = T * random_phase  # Add random phase to the target

                A = np.fft.ifft2(np.fft.ifftshift(E))  # Compute the backward propagation from the target to hologram plane

                H = np.angle(A)  # Get the phase of hologram
                H = np.double(H > 0)  # Binary phase quantization

                HoloPerChannel = HoloPerChannel + H * 2 ** (
                            PerChannelFrameIndex - 1)  # Encode the hologram bit-plane by bit-plane

                R = np.abs(np.fft.fftshift(np.fft.fft2(np.fft.fftshift(np.exp(1j * H * np.pi)))))  # Calculate total reconstruction
                R = R / np.sqrt(np.sum(np.abs(R) ** 2) / np.sum(np.abs(TargetAmplitude) ** 2))

                R_total = R_total + R
                running = R_total / holo_frame_i / np.max(TargetAmplitude)

            FreemanHolo[:, :, HoloChannelIndex - 1] = HoloPerChannel

        avg_recon = R_total / holo_frame_i / np.max(TargetAmplitude)

        return avg_recon, FreemanHolo
        

    def gradient_descent(
            self,
            loss_function,
            n_iter: int = 1_000,
            lr: float = 1e-2,
            roi: tuple = (slice(None), slice(None)),
            finite_eps: float = 1e-4,
            sample_ratio: float = 0.10
    ):

        image = self.image_array.image_array
        TargetAmplitude = np.sqrt(image.astype(np.float32))

        # continuous phase ∈ [0,1)
        H_float = np.random.rand(*TargetAmplitude.shape).astype(np.float32)

        loss_trace = []
        h, w = TargetAmplitude.shape
        hw = h * w

        # pre-detect loss type by name (cheap heuristic)
        for _ in tqdm(range(n_iter), disable=not self.verbose, desc="GD"):

            #forward model
            field = np.exp(1j * np.pi * H_float)
            R = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(field)))
            A = np.abs(R)

            scale = np.sqrt((TargetAmplitude ** 2).sum() / (A ** 2).sum())
            A *= scale

            A_roi = A[roi]
            T_roi = TargetAmplitude[roi]

            # loss + gradient wrt amplitude

            L, g_roi = loss_function(A_roi, T_roi)
            g_A = np.zeros_like(A, dtype=np.float32)
            g_A[roi] = g_roi

            loss_trace.append(L)

            # back-prop to phase 
            denom = A + 1e-12
            g_Q = g_A * (R / denom)
            g_f = hw * np.fft.ifft2(np.fft.ifftshift(g_Q))
            g_f = np.fft.fftshift(g_f)
            g_phase = np.pi * np.imag(g_f * np.conj(field))

            # phase update & wrap 
            H_float -= lr * g_phase.astype(np.float32)
            H_float %= 1.0

        # binarise & final reconstruction 
        H_bin = (H_float > 0.5).astype(np.float32)
        R_final = np.fft.fftshift(
            np.fft.fft2(np.fft.fftshift(np.exp(1j * H_bin * np.pi))))
        R_final = np.abs(R_final)
        R_final *= np.sqrt((TargetAmplitude ** 2).sum() / (R_final ** 2).sum())
        print(H_bin)

        return R_final.astype(np.float32), loss_trace

    def gradient_descent_cont(
            self,
            loss_function,  # unchanged signature
            n_iter: int = 1_000,
            lr: float = 5e-3,
            roi: tuple = (slice(None), slice(None)),
            finite_eps: float = 1e-4,
            sample_ratio: float = 0.10
    ):

        image = self.image_array.image_array
        TargetAmplitude = np.sqrt(image)
        TargetAmplitude = np.nan_to_num(image, nan=0.0, posinf=np.max(image), neginf=0.0) # may be uneccasary

        # continuous phase ∈ [0,1)  (same initialisation)
        H_float = np.random.rand(*TargetAmplitude.shape).astype(np.float32)

        loss_trace = []
        h, w = TargetAmplitude.shape
        hw = h * w

        for _ in tqdm(range(n_iter), disable=not self.verbose, desc="GD"):
            # forward model
            # CHANGED:   pi  →  2pi   (full continuous phase range)
            field = np.exp(1j * 2 * np.pi * H_float)

            R = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(field)))
            A = np.abs(R)

            scale = np.sqrt((TargetAmplitude ** 2).sum() / (A ** 2).sum())
            A *= scale

            A_roi = A[roi]
            T_roi = TargetAmplitude[roi]

            # loss + gradient wrt amplitude 
            L, g_roi = loss_function(A_roi, T_roi)  # no change
            g_A = np.zeros_like(A, dtype=np.float32)
            g_A[roi] = g_roi
            loss_trace.append(L)

            # back-prop to phase 
            denom = A + 1e-12
            g_Q = g_A * (R / denom)
            g_f = hw * np.fft.ifft2(np.fft.ifftshift(g_Q))
            g_f = np.fft.fftshift(g_f)

            # CHANGED:  gradient factor pi  →  2pi
            g_phase = 2 * np.pi * np.imag(g_f * np.conj(field))

            # phase update & wrap 
            H_float -= lr * g_phase.astype(np.float32)
            H_float %= 1.0  # keep 0…1 (⇒ 0…2pi rad)

        # final reconstruction  (NO binarisation)
        # CHANGED:  remove threshold; use continuous phase directly
        field_c = np.exp(1j * 2 * np.pi * H_float)
        R_final = np.fft.fftshift(
            np.fft.fft2(np.fft.fftshift(field_c)))
        R_final = np.abs(R_final)
        R_final *= np.sqrt((TargetAmplitude ** 2).sum() / (R_final ** 2).sum())

        return R_final.astype(np.float32), loss_trace, H_float



