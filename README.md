# Synthetic MRI Brain Tumor Data Generation using GANs

Bachelor's thesis project exploring three GAN architectures for generating synthetic MRI brain tumor images to augment limited medical datasets.

## Architecture

```
                        +-------------------+
  Random noise z -----> |    Generator G    | -----> Fake MRI image (64x64x3)
  (100-dim vector)      | ConvTranspose2d x4|               |
                        +-------------------+               |
                                                            v
  Real MRI image -----> +-------------------+        +-------------+
  (from Dataset/)       |  Discriminator D  | -----> | Real / Fake |
                        |    Conv2d x4      |        +-------------+
                        +-------------------+
```

Three variants implemented:
- **DCGAN** -- unconditional generation, no class information
- **CGAN** -- class label concatenated with noise vector, generator produces class-specific images
- **ACGAN** -- auxiliary classifier head on discriminator predicts tumor type alongside real/fake

## Dataset

[Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) from Kaggle. 4 classes:

| Class      | Samples | Description              |
|------------|---------|--------------------------|
| Glioma     | 1,621   | Glial cell tumor         |
| Meningioma | 1,645   | Meninges tumor           |
| Pituitary  | 1,757   | Pituitary gland tumor    |
| No Tumor   | 2,000   | Healthy brain MRI        |

Not included in this repo due to size. Download and place in `Dataset/`.

## Quick Start

### Notebooks (Google Colab)

Upload the notebooks to Colab and run all cells. The DCGAN notebook is self-contained.

### CLI Training

```bash
pip install -r requirements.txt

python train.py --dataroot Dataset --epochs 10 --lr 0.0002 --batch_size 64
```

### Docker

```bash
docker build -t mri-gan .
docker run --gpus all -v ./Dataset:/app/Dataset -v ./output:/app/output mri-gan
```

## Training Results (DCGAN, 10 epochs)

```
[0/10][0/110]   Loss_D: 2.1370  Loss_G: 3.8727  D(x): 0.4169  D(G(z)): 0.5891
[1/10][0/110]   Loss_D: 0.3692  Loss_G: 5.3744  D(x): 0.8018  D(G(z)): 0.0145
...
[9/10][100/110] Loss_D: 0.3920  Loss_G: 5.0865  D(x): 0.8647  D(G(z)): 0.1544
```

D(x) stabilized around 0.86 (discriminator correctly identifies ~86% of real images). Generator loss plateaued around epoch 7, suggesting longer training or learning rate scheduling could improve results.

## Structure

```
.
├── train.py                  # Standalone DCGAN training script (CLI)
├── utils.py                  # Helper functions (image saving, weight init)
├── MRI_DCGAN.ipynb           # DCGAN notebook
├── MRI_CGAN.ipynb            # Conditional GAN notebook
├── ACGAN.ipynb               # Auxiliary Classifier GAN notebook
├── BrainTumorMRI_using_DnseNet.ipynb  # DenseNet classifier for comparison
├── training_log_dcgan.txt    # Training output log
├── requirements.txt
└── Dockerfile
```

## Requirements

```
pip install -r requirements.txt
```

- Python 3.10+
- PyTorch >= 2.0
- torchvision, numpy, matplotlib, imageio, scipy

## References

- Radford et al., "Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks" (2015)
- Mirza & Osindero, "Conditional Generative Adversarial Nets" (2014)
- Odena et al., "Conditional Image Synthesis with Auxiliary Classifier GANs" (2017)
- Nickparvar, M., "Brain Tumor MRI Dataset", Kaggle
