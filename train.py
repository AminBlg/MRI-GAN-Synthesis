"""
Standalone DCGAN training script for MRI brain tumor image synthesis.
Extracted from MRI_DCGAN.ipynb for reproducible CLI-based training.

Usage:
    python train.py --dataroot Dataset --epochs 10 --lr 0.0002 --batch_size 64
"""

import argparse
import os
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

from utils import initialize_weights


class Generator(nn.Module):
    def __init__(self, nz, ngf, nc):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)


class Discriminator(nn.Module):
    def __init__(self, nc, ndf):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)


def train(args):
    # Seed for reproducibility
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data pipeline
    dataset = dset.ImageFolder(
        root=args.dataroot,
        transform=transforms.Compose([
            transforms.Resize(args.image_size),
            transforms.CenterCrop(args.image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=args.workers
    )
    print(f"Dataset: {len(dataset)} images, {len(dataloader)} batches")

    # Models
    netG = Generator(args.nz, args.ngf, args.nc).to(device)
    netG.apply(initialize_weights)

    netD = Discriminator(args.nc, args.ndf).to(device)
    netD.apply(initialize_weights)

    criterion = nn.BCELoss()
    fixed_noise = torch.randn(64, args.nz, 1, 1, device=device)

    real_label = 1.0
    fake_label = 0.0

    optimizerD = optim.Adam(netD.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("Starting training...")
    G_losses = []
    D_losses = []

    with open(os.path.join(args.output_dir, "training_log.txt"), "w") as log_file:
        for epoch in range(args.epochs):
            for i, data in enumerate(dataloader, 0):
                # Update Discriminator
                netD.zero_grad()
                real_cpu = data[0].to(device)
                b_size = real_cpu.size(0)
                label = torch.full((b_size,), real_label, dtype=torch.float, device=device)

                output = netD(real_cpu).view(-1)
                errD_real = criterion(output, label)
                errD_real.backward()
                D_x = output.mean().item()

                noise = torch.randn(b_size, args.nz, 1, 1, device=device)
                fake = netG(noise)
                label.fill_(fake_label)
                output = netD(fake.detach()).view(-1)
                errD_fake = criterion(output, label)
                errD_fake.backward()
                D_G_z1 = output.mean().item()
                errD = errD_real + errD_fake
                optimizerD.step()

                # Update Generator
                netG.zero_grad()
                label.fill_(real_label)
                output = netD(fake).view(-1)
                errG = criterion(output, label)
                errG.backward()
                D_G_z2 = output.mean().item()
                optimizerG.step()

                if i % 50 == 0:
                    line = (
                        f"[{epoch}/{args.epochs}][{i}/{len(dataloader)}]\t"
                        f"Loss_D: {errD.item():.4f}\tLoss_G: {errG.item():.4f}\t"
                        f"D(x): {D_x:.4f}\tD(G(z)): {D_G_z1:.4f} / {D_G_z2:.4f}"
                    )
                    print(line)
                    log_file.write(line + "\n")
                    log_file.flush()

                G_losses.append(errG.item())
                D_losses.append(errD.item())

            # Save sample images each epoch
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            vutils.save_image(
                fake, os.path.join(args.output_dir, f"fake_epoch_{epoch:03d}.png"),
                normalize=True, nrow=8
            )

    # Save weights
    torch.save(netG.state_dict(), os.path.join(args.output_dir, "netG_weights.pth"))
    torch.save(netD.state_dict(), os.path.join(args.output_dir, "netD_weights.pth"))
    print(f"Training complete. Weights saved to {args.output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DCGAN training for MRI brain tumor synthesis")
    parser.add_argument("--dataroot", type=str, default="Dataset", help="Root directory for dataset")
    parser.add_argument("--workers", type=int, default=2, help="Number of data loading workers")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size during training")
    parser.add_argument("--image_size", type=int, default=64, help="Spatial size of training images")
    parser.add_argument("--nc", type=int, default=3, help="Number of channels in training images")
    parser.add_argument("--nz", type=int, default=100, help="Size of latent vector")
    parser.add_argument("--ngf", type=int, default=64, help="Size of feature maps in generator")
    parser.add_argument("--ndf", type=int, default=64, help="Size of feature maps in discriminator")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.0002, help="Learning rate for Adam")
    parser.add_argument("--beta1", type=float, default=0.5, help="Beta1 for Adam optimizer")
    parser.add_argument("--output_dir", type=str, default="output", help="Output directory for weights and samples")
    parser.add_argument("--seed", type=int, default=999, help="Random seed for reproducibility")
    args = parser.parse_args()
    train(args)
