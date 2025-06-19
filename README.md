# Computer generated holography Sandbox

This repository serves as a sandbox environment for experimenting with computational holography (CGH), particularly focused on testing and developing new hologram generation algorithms and custom loss functions. It is designed for rapid prototyping and evaluation of ideas related to phase retrieval, wavefront modulation, and 3D light field reconstruction.

## Background

Computational Holography (CGH) involves generating holograms using numerical methods to simulate light propagation and interference. Unlike traditional optical holography, CGH enables precise control over the phase and amplitude of light using algorithms, often leveraging spatial light modulators (SLMs) or digital micromirror devices (DMDs).

Advances in CGH have been driven by new algorithms (e.g., Gerchberg–Saxton, ADMM, deep learning-based methods) and the design of loss functions that better capture perceptual quality, depth fidelity, or energy efficiency. This repository provides a modular testing environment to prototype such approaches.

## Features

Modular pipeline for testing new hologram generation algorithms. Currently provided are Simulated Annealing, Gradient Descent, One Step Phase Retrieval, Genetic Algorithm, and Direct Binary Search. These algorithms have been implemented using a NumPy only approach with low external dependencies to allow for simple editing. 

Custom loss function integration for perceptual and task-specific optimization. Currently provided options include Mean Squared Error, Peak Signal to Noise Ratio, Structural Similarity Index Measure, Wavelet Normalised Cross Correlation. See S.G. publications for ongoing loss functions research. 

Support for phase-only and amplitude-modulated holograms (not comprehensive, see notes on cont and binary phase). 

Utilities for simulating propagation (Fourier).

http://www-g.eng.cam.ac.uk/CMMPE/
