import os
import torch.nn as nn
import numpy as np
import imageio
import matplotlib.pyplot as plt


def print_network(net):
    """Print network architecture and parameter count."""
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


def save_images(images, size, image_path):
    """Save a grid of images to disk."""
    image = np.squeeze(merge(images, size))
    return imageio.imwrite(image_path, image)


def merge(images, size):
    """Merge multiple images into a single grid."""
    h, w = images.shape[1], images.shape[2]
    if images.shape[3] in (3, 4):
        c = images.shape[3]
        img = np.zeros((h * size[0], w * size[1], c))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = image
        return img
    elif images.shape[3] == 1:
        img = np.zeros((h * size[0], w * size[1]))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w] = image[:, :, 0]
        return img
    else:
        raise ValueError(
            'in merge(images,size) images parameter '
            'must have dimensions: HxW or HxWx3 or HxWx4'
        )


def loss_plot(hist, path='Train_hist.png', model_name=''):
    """Plot discriminator and generator loss curves."""
    x = range(len(hist['D_loss']))
    plt.plot(x, hist['D_loss'], label='D_loss')
    plt.plot(x, hist['G_loss'], label='G_loss')
    plt.xlabel('Iter')
    plt.ylabel('Loss')
    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()
    path = os.path.join(path, model_name + '_loss.png')
    plt.savefig(path)
    plt.close()


def initialize_weights(net):
    """Initialize network weights following DCGAN paper conventions."""
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
        elif isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.02)
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.zero_()
