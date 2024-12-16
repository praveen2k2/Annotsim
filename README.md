---
# Annotsim: Self-Supervised Anomaly Segmentation via Diffusion Models with Dynamic Transformer UNet (WACV 2024) 🚀

This research paper introduces a novel self-supervised anomaly detection method for image segmentation. It employs a diffusion model utilizing a newly developed four-dimensional simplex noise function (**Tsimplex**) for improved efficiency and sample quality, especially in higher-dimensional and colored images. The core of the model is a **Dynamic Transformer UNet (DTUNet)**, a modified Vision Transformer architecture designed to handle time and noise image patches as tokens. Extensive experiments across three datasets demonstrate significant performance improvements over existing generative-based anomaly detection methods, particularly in medical imaging. The source code is publicly available.

## Features

- **Custom Diffusion Models**: An extended UNet with selective denoising capabilities.
- **Dynamic Transformer Blocks**: Enhancements for dynamic anomaly segmentation.
- **Simplex Noise Integration**: Supports 3D/4D noise generation for improved feature diversity.
- **Comprehensive Evaluation Metrics**: Tools to calculate precision, recall, Dice score, and more.
- **Visualization Tools**: Generates diffusion videos and detection outputs for interpretability.

---

## Example Outputs

### Diffusion Training Visualization
![Distribution Visualization](assets/Timed_simplex_histogram1.png) 
![Histogram Plot](assets/Timed_simplex_histogram1.png)
![Octave Visualization](assets/SIMPLEX_TEST_Oct.gif)
![SSIM Plot](assets/SSIM_plot.pdf)
![Time Complexity Plots](assets/time_complexity_plot.pdf)
![MRI Translation](assets/results/args200/Generation/1000_500_No22.png) 

### Anomaly Detection Results
![Anomaly Detection](assets/anomaly_detection_example.png)

### Simplex Noise Example
![Simplex Noise](assets/simplex_noise_example.png)

---

## Project Structure

```plaintext
Annotsim/
├── src/
│   ├── models/               # Model architectures (UNet, Transformer blocks, etc.)
│   ├── utils/                # Helper functions (dataset loading, noise generation, etc.)
│   ├── scripts/              # Training and evaluation scripts
├── assets/                   # Visual assets and results
├── requirements.txt          # Python dependencies
├── setup.py                  # Installation script
├── README.md                 # Project documentation
└── .gitignore                # Ignored files and directories
