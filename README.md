# Self-Supervised Anomaly Segmentation via Diffusion Models with Dynamic Transformer UNet (WACV 2024)

This repository contains the implementation of an anomaly detection approach utilizing Denoising Diffusion Probabilistic Models (DDPMs) with simplex noise, developed in PyTorch. The work focuses on leveraging diffusion models for self-supervised anomaly segmentation.

The code was written by [Komal Kumar](https://github.com/MAXNORM8650) and draws inspiration from the following repositories:
- [Denoising Diffusion Model for Anomaly Detection](https://github.com/Julian-Wyatt/AnoDDPM)
- [Predictive Convolutional Attentive Block](https://github.com/ristea/sspcab)
- [Guided Diffusion](https://github.com/openai/guided-diffusion)

---

## Features

- **Custom Diffusion Models**: Extended UNet with selective denoising for better anomaly segmentation.
- **Simplex Noise Generator**: Supports 3D/4D noise generation for enhanced applications.
- **Comprehensive Evaluation**: Includes scripts for precision, recall, and Dice score calculation.
- **Visualization Tools**: Generates diffusion videos and detection images for better interpretability.

---

## Example Outputs

### Diffusion Training Visualization
![Diffusion Training](assets/diffusion_training_example.png)

### Anomaly Detection Results
![Anomaly Detection](assets/anomaly_detection_example.png)

### Simplex Noise Example
![Simplex Noise](assets/simplex_noise_example.png)

---

## Repository Structure

### Core Files
- **`dataset.py`**: Custom dataset loader for training and testing.
- **`detection.py`**: Handles detection metrics and experimentation.
- **`diffusion_training.py`**: Implements the training loop for diffusion models.
- **`evaluation.py`**: Functions for evaluating model performance.
- **`GaussianDiffusion.py`**: Implements Gaussian architecture for diffusion models.
- **`helpers.py`**: Utility functions for operations like checkpoint handling.
- **`simplex.py`**: A simplex noise generator extended for 3D/4D noise.
- **`UNet.py`**: Implements a custom UNet with selective denoising blocks.

### Key Directories
- **`test_args/`**: Contains JSON configuration files for testing.
- **`model/diff-params-ARGS={i}/`**: Checkpoints saved during training.
- **`diffusion-videos/ARGS={i}/`**: Video outputs generated during training and testing.
- **`diffusion-training-images/ARGS={i}/`**: Images generated during diffusion training and detection.

---

## Installation

### Prerequisites
- Python 3.8 or later
- CUDA-enabled GPU (optional but recommended for training)

### Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/MAXNORM8650/Annotsim.git
   cd Annotsim
