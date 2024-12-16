Here’s the modified version of your README that reduces the size and improves formatting for visual clarity while keeping all the relevant information:

Annotsim: Self-Supervised Anomaly Segmentation via Diffusion Models with Dynamic Transformer UNet (WACV 2024) 📈

This repository presents Annotsim, a novel self-supervised anomaly detection method for image segmentation. It utilizes a diffusion model with the Tsimplex noise function, optimized for high-dimensional and colored images. At its core is a Dynamic Transformer UNet (DTUNet), a modified Vision Transformer capable of handling time and noise image patches as tokens. The approach achieves state-of-the-art performance in medical imaging datasets.

Key Features
	•	Custom Diffusion Models: Incorporates selective denoising with extended UNet.
	•	Dynamic Transformer Blocks: Enhances anomaly segmentation with dynamic adaptability.
	•	Simplex Noise Integration: Supports advanced noise generation for improved diversity.
	•	Visualization Tools: Outputs diffusion videos, detection results, and interpretability plots.
	•	Metrics: Built-in tools for computing precision, recall, and Dice scores.

Example Outputs

Diffusion Model Visualizations

Anomaly Detection Results

Noise and Complexity Plots

Simplex Noise Example	SSIM Plot
	

Time Complexity Plot	MRI Translation Example
	

Project Structure

Annotsim/
├── src/
│   ├── models/               # Model architectures (UNet, Transformer blocks, etc.)
│   ├── utils/                # Dataset loading, noise generation, etc.
│   ├── scripts/              # Training and evaluation scripts
├── requirements.txt          # Python dependencies
├── setup.py                  # Installation script
├── README.md                 # Project documentation
└── .gitignore                # Ignored files and directories

Installation

Prerequisites
	•	Python 3.8 or later
	•	CUDA-enabled GPU (recommended for training)

Steps
	1.	Clone the repository:

git clone https://github.com/MAXNORM8650/Annotsim.git
cd Annotsim


	2.	Install dependencies:

pip install -r requirements.txt


	3.	Install the repository as a package:

pip install -e .

Usage

Train a Diffusion Model

python src/scripts/diffusion_training_UVW.py --argN

Replace argN with your configuration from the test_args directory.

Evaluate a Model

python src/scripts/detection.py --argN

Datasets

This project uses the following datasets:
	1.	BRATS2021: Brain tumor segmentation.
	2.	Pneumonia X-Ray: Chest X-ray anomaly detection.

Refer to the datasets’ official documentation for more details.

Results

Outputs
	1.	Diffusion Videos: Saved in outputs/diffusion-videos/.
	2.	Detection Results: Saved in outputs/detection-images/.

Citation

If you use this work, please cite:

@inproceedings{kumar2023self,
  title={Self-supervised Diffusion Model for Anomaly Segmentation in Medical Imaging},
  author={Kumar, Komal and Chakraborty, Snehashis and Roy, Sudipta},
  booktitle={International Conference on Pattern Recognition and Machine Intelligence},
  pages={359--368},
  year={2023},
  organization={Springer}
}

Contributors
	•	Komal Kumar: GitHub Profile

Acknowledgments

This project is based on ideas from:
	•	AnoDDPM
	•	Predictive Convolutional Attentive Block
	•	Guided Diffusion

Improvements
	1.	Simplified Output Figures: Grouped images/tables for better readability.
	2.	Clean Structure: Project structure and commands organized for clarity.
	3.	Reduced Repetition: Removed redundant figure mentions.

You can update the README directly in the GitHub repo or clone and modify locally. Let me know if you need further adjustments! 🚀
