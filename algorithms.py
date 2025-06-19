import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import PIL
import skimage
from skimage.transform import resize
# from skimage.metrics import mean_squared_error
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
        #TargetAmplitude = np.sqrt(image.image_array)
        TargetAmplitude = np.sqrt(image)
        H = np.round(np.random.rand(*TargetAmplitude.shape))
        R = abs(np.fft.fftshift(np.fft.fft2(np.fft.fftshift(np.exp(1j * H * np.pi)))))
        R = abs(R) * np.sqrt(np.sum(np.sum(TargetAmplitude ** 2)) / np.sum(np.sum(np.abs(R) ** 2)))
        L = loss_function(R[roi], TargetAmplitude[roi])

        loss_list = []  # Record loss (for plotting)
        # for n in range(n_iter):
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
        #H_list = [np.round(np.random.rand(*T.shape)) for _ in range(N)]
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

        #N = 50  # Number of iterations
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

    def gradient_descent_old(
            self,
            loss_function,
            n_iter: int = 1_000,
            lr: float = 1e-2,
            roi: tuple = (slice(None), slice(None)),
            finite_eps: float = 1e-4,  # FD step for ∂L/∂A
            sample_ratio: float = 0.10,  # stochastic FD for speed
    ):

        image = self.image_array.image_array
        TargetAmplitude = np.sqrt(image.astype(np.float32))

        # --------  continuous phase ϕ ∈ [0,1)  ----------------------
        H_float = np.random.rand(*TargetAmplitude.shape).astype(np.float32)

        loss_trace = []
        h, w = TargetAmplitude.shape
        hw = h * w

        # Pre-compute ROI mask & linear indices for stochastic FD
        roi_mask = np.zeros_like(TargetAmplitude, dtype=bool)
        roi_mask[roi] = True
        roi_indices = np.argwhere(roi_mask)  # (N_roi, 2)

        for _ in tqdm(range(n_iter), disable=not self.verbose, desc="GD"):

            # 1. Forward model  ---------------- ϕ → amplitude A
            field = np.exp(1j * np.pi * H_float)
            R = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(field)))
            A = np.abs(R)

            # energy normalisation (same formula as DBS)
            scale = np.sqrt((TargetAmplitude ** 2).sum() / (A ** 2).sum())
            A *= scale

            # Compute scalar loss only (no gradient yet)
            L = loss_function(A[roi], TargetAmplitude[roi])
            loss_trace.append(float(L))

            # 2. Finite-difference ∂L/∂A on a random 10 % ROI subset
            g_A = np.zeros_like(A, dtype=np.float32)

            # stochastic subset for speed
            sample_idx = roi_indices[
                np.random.choice(len(roi_indices),
                                 size=int(sample_ratio * len(roi_indices)),
                                 replace=False)
            ]

            for (y, x) in sample_idx:
                A_perturb = A[y, x]
                A[y, x] += finite_eps
                L_plus = loss_function(A[roi], TargetAmplitude[roi])
                g_A[y, x] = (L_plus - L) / finite_eps
                A[y, x] = A_perturb  # restore

            # 3. Back-propagate to phase ϕ analytically
            denom = A + 1e-12
            g_Q = g_A * (R / denom)  # ∂|Q| wrt Q
            g_field = hw * np.fft.ifft2(np.fft.ifftshift(g_Q))
            g_field = np.fft.fftshift(g_field)
            g_phase = np.pi * np.imag(g_field * np.conj(field))

            # 4. Gradient-descent update & wrap to [0,1)
            H_float -= lr * g_phase.astype(np.float32)
            H_float %= 1.0

        # 5. Binarise & final reconstruction
        H_bin = (H_float > 0.5).astype(np.float32)
        R_final = np.fft.fftshift(
            np.fft.fft2(np.fft.fftshift(np.exp(1j * H_bin * np.pi))))
        R_final = np.abs(R_final)
        R_final *= np.sqrt((TargetAmplitude ** 2).sum() / (R_final ** 2).sum())

        return R_final.astype(np.float32), loss_trace

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

        # continuous phase ϕ ∈ [0,1)
        H_float = np.random.rand(*TargetAmplitude.shape).astype(np.float32)

        loss_trace = []
        h, w = TargetAmplitude.shape
        hw = h * w

        # pre-detect loss type by name (cheap heuristic)
        for _ in tqdm(range(n_iter), disable=not self.verbose, desc="GD"):

            # ---------- forward model ----------------------------------
            field = np.exp(1j * np.pi * H_float)
            R = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(field)))
            A = np.abs(R)

            scale = np.sqrt((TargetAmplitude ** 2).sum() / (A ** 2).sum())
            A *= scale

            A_roi = A[roi]
            T_roi = TargetAmplitude[roi]

            # ---------- loss + gradient wrt amplitude ------------------

            L, g_roi = loss_function(A_roi, T_roi)
            g_A = np.zeros_like(A, dtype=np.float32)
            g_A[roi] = g_roi

            loss_trace.append(L)

            # ---------- back-prop to phase ϕ ---------------------------
            denom = A + 1e-12
            g_Q = g_A * (R / denom)
            g_f = hw * np.fft.ifft2(np.fft.ifftshift(g_Q))
            g_f = np.fft.fftshift(g_f)
            g_phase = np.pi * np.imag(g_f * np.conj(field))

            # ---------- phase update & wrap ----------------------------
            H_float -= lr * g_phase.astype(np.float32)
            H_float %= 1.0

        # ---------- binarise & final reconstruction -------------------
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
        #TargetAmplitude = np.sqrt(image.astype(np.float32))
        TargetAmplitude = np.sqrt(image)
        TargetAmplitude = np.nan_to_num(image, nan=0.0, posinf=np.max(image), neginf=0.0) # may be uneccasary

        # continuous phase ϕ ∈ [0,1)  (same initialisation)
        H_float = np.random.rand(*TargetAmplitude.shape).astype(np.float32)

        loss_trace = []
        h, w = TargetAmplitude.shape
        hw = h * w

        for _ in tqdm(range(n_iter), disable=not self.verbose, desc="GD"):
            # ---------- forward model ----------------------------------
            # ★ CHANGED:   π  →  2π   (full continuous phase range)
            field = np.exp(1j * 2 * np.pi * H_float)

            R = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(field)))
            A = np.abs(R)

            scale = np.sqrt((TargetAmplitude ** 2).sum() / (A ** 2).sum())
            A *= scale

            A_roi = A[roi]
            T_roi = TargetAmplitude[roi]

            # ---------- loss + gradient wrt amplitude ------------------
            L, g_roi = loss_function(A_roi, T_roi)  # no change
            g_A = np.zeros_like(A, dtype=np.float32)
            g_A[roi] = g_roi
            loss_trace.append(L)

            # ---------- back-prop to phase ϕ ---------------------------
            denom = A + 1e-12
            g_Q = g_A * (R / denom)
            g_f = hw * np.fft.ifft2(np.fft.ifftshift(g_Q))
            g_f = np.fft.fftshift(g_f)

            # ★ CHANGED:  gradient factor π  →  2π
            g_phase = 2 * np.pi * np.imag(g_f * np.conj(field))

            # ---------- phase update & wrap ----------------------------
            H_float -= lr * g_phase.astype(np.float32)
            H_float %= 1.0  # keep 0…1 (⇒ 0…2π rad)



        # ---------- final reconstruction  (NO binarisation) -----------
        # ★ CHANGED:  remove threshold; use continuous phase directly
        field_c = np.exp(1j * 2 * np.pi * H_float)
        R_final = np.fft.fftshift(
            np.fft.fft2(np.fft.fftshift(field_c)))
        R_final = np.abs(R_final)
        R_final *= np.sqrt((TargetAmplitude ** 2).sum() / (R_final ** 2).sum())

        return R_final.astype(np.float32), loss_trace, H_float

    def genetic_algorithm_opt(
            self,
            crossover_function,
            parent_selection_function,
            N=5000,
            N_iter=1000,
            roi=(slice(None), slice(None)),
            backend: str = "numpy",  # 'numpy', 'cupy', or 'pyfftw'
            mutation_rate: float = 0.02,
    ):
        # ---------------- choose array module -------------------------
        xp = np

        # ---------------- initial data & constants -------------------
        T = xp.asarray(np.sqrt(self.image_array.image_array), dtype=xp.float32)
        E_T = float((T ** 2).sum())  # scalar, CPU OK
        H = xp.random.rand(N, *T.shape).astype(xp.float32)  # population tensor

        error_trace, R_best, H_best = [], None, None
        lowest_mse = float("inf")

        # ---------------- helper: forward model for a batch ----------
        def forward(h_batch):
            field = xp.exp(1j * xp.pi * h_batch)
            R = xp.fft.fftshift(xp.fft.fft2(xp.fft.fftshift(field, axes=(-2, -1)),
                                            norm=None, axes=(-2, -1)), axes=(-2, -1))
            absR = xp.abs(R)
            # energy normalisation per individual
            E_R = xp.sum(absR ** 2, axis=(-2, -1), keepdims=True)
            scale = xp.sqrt(E_T / E_R)
            return absR * scale

        # ---------------- GA main loop -------------------------------
        for it in tqdm(range(N_iter), disable=not self.verbose, desc="GA"):

            R = forward(H)
            mse = ((R[..., roi[0], roi[1]] - T[roi]) ** 2).mean(axis=(-2, -1)) / (
                (T[roi] ** 2).sum())  # shape (N,)

            # track best
            idx_min = int(xp.argmin(mse))
            if float(mse[idx_min]) < lowest_mse:
                lowest_mse = float(mse[idx_min])
                R_best = R[idx_min]
                H_best = H[idx_min]

            error_trace.append(float(mse.mean()))

            # --- selection & crossover on CPU-friendly arrays --------
            mse_cpu = np.asarray(mse)
            H_cpu = np.asarray(H)
            parents = parent_selection_function(mse_cpu, H_cpu)
            children = crossover_function(parents, n_splits=4)

            # # optional mutation (continuous)
            # mask = np.random.rand(*children.shape) < mutation_rate
            # children = (children + 0.05 * np.random.randn(*children.shape) * mask) % 1.0

            # keep population size N
            H = xp.asarray(children[:N], dtype=xp.float32)

            # early stop criterion
            if len(error_trace) >= 4 and len(set(np.round(error_trace[-4:], 8))) == 1:
                break

        # ---------------- final reconstructions ----------------------
        R_final_batch = forward(H)
        R_list = [r for r in R_final_batch]

        return R_best, error_trace, H_best, np.asarray(H_cpu), R_list

    # def pad_to_square_complex(arr: np.ndarray):
    #     H, W = arr.shape
    #     S = max(H, W)
    #     pad_top    = (S - H) // 2
    #     pad_bottom = S - H - pad_top
    #     pad_left   = (S - W) // 2
    #     pad_right  = S - W - pad_left
    #     return np.pad(arr,
    #                   pad_width=((pad_top, pad_bottom),
    #                              (pad_left, pad_right)),
    #                   mode="constant",
    #                   constant_values=0), (pad_top, pad_left)



    # ──────────────────────────────────────────────────────────────
    # NEW version of the continuous-phase gradient descent
    # ──────────────────────────────────────────────────────────────
    def gradient_descent_cont_new(
            self,
            loss_function,              # unchanged signature
            n_iter: int = 1_000,
            lr: float = 5e-3,
            roi: tuple = (slice(None), slice(None)),
            finite_eps: float = 1e-4,   # kept for signature compatibility
            sample_ratio: float = 0.10  # kept for signature compatibility
    ):
        # ---------- target amplitude -----------------------------------------
        image = self.image_array.image_array
        TargetAmplitude = np.sqrt(image).astype(np.float32)
        TargetAmplitude = np.nan_to_num(TargetAmplitude, nan=0.0)

        # pad the target so that *field* and *target* live on the same square
        H, W = TargetAmplitude.shape
        TargetAmplitude_sq, (pad_top, pad_left) = pad_to_square_complex(TargetAmplitude)
        S = TargetAmplitude_sq.shape[0]          # side length of the square grid

        # ROI must be shifted into the padded coordinates
        tgt_roi = (slice(pad_top,  pad_top + H),
                   slice(pad_left, pad_left + W))

        # ---------- initialise phase -----------------------------------------
        H_float = np.random.rand(H, W).astype(np.float32)  # ϕ in [0,1)

        loss_trace = []
        hw = S * S                                         # for the gradient scale

        for _ in tqdm(range(n_iter), disable=not self.verbose, desc="GD"):
            # forward model on the square grid
            field = np.exp(1j * 2 * np.pi * H_float)
            field, _ = pad_to_square_complex(field)        # (S, S)

            R = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(field)))
            A = np.abs(R)

            # energy normalisation
            scale = np.sqrt((TargetAmplitude_sq**2).sum() / (A**2).sum())
            A *= scale

            # loss on the rectangular content area only
            A_roi = A[tgt_roi]
            T_roi = TargetAmplitude_sq[tgt_roi]

            L, g_roi = loss_function(A_roi, T_roi)
            g_A = np.zeros_like(A, dtype=np.float32)
            g_A[tgt_roi] = g_roi
            loss_trace.append(L)

            # back-propagate gradient to phase
            denom = A + 1e-12
            g_Q = g_A * (R / denom)
            g_f = hw * np.fft.ifft2(np.fft.ifftshift(g_Q))
            g_f = np.fft.fftshift(g_f)
            g_phase = 2 * np.pi * np.imag(g_f * np.conj(field))

            # update & wrap phase (only on the un-padded support)
            g_phase_rect = g_phase[tgt_roi]                # crop to H×W
            H_float -= lr * g_phase_rect.astype(np.float32)
            H_float %= 1.0

        # ---------- final reconstruction -------------------------------------
        field_c, _ = pad_to_square_complex(np.exp(1j * 2 * np.pi * H_float))
        R_final = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(field_c)))
        R_final = np.abs(R_final)
        R_final *= np.sqrt((TargetAmplitude_sq**2).sum() / (R_final**2).sum())

        # crop back to original rectangle for metrics / display
        R_final_rect = R_final[tgt_roi]

        return R_final_rect.astype(np.float32), loss_trace, H_float



