# Synthetic MRI Brain Tumor Data Generation using GANs

Bachelor's project exploring three GAN architectures for generating synthetic MRI brain tumor images:

- **DCGAN** (Deep Convolutional GAN) — `MRI_DCGAN.ipynb`
- **CGAN** (Conditional GAN) — `MRI_CGAN.ipynb`
- **ACGAN** (Auxiliary Classifier GAN) — `ACGAN.ipynb`

Built with PyTorch. Also includes a DenseNet-based classifier for comparison (`BrainTumorMRI_using_DnseNet.ipynb`).

## Dataset

Brain Tumor MRI Dataset from Kaggle — 4 classes: glioma, meningioma, pituitary, no tumor. Not included in this repo due to size. Download from [Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) and place in `Dataset/`.

## Requirements

- Python 3.10+
- PyTorch
- torchvision
- matplotlib
- numpy
