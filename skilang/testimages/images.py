import random
import struct
from typing import Tuple, List

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from tests.resources.datasets import dataset_path


def load_images(filename):
    with open(filename, 'rb') as f:
        magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return torch.tensor(data, dtype=torch.uint8).view(num, 28, 28)

def load_labels(filename):
    with open(filename, 'rb') as f:
        magic, num = struct.unpack('>II', f.read(8))
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return torch.tensor(data, dtype=torch.long)

def preprocess_images(images):
    images = images.float() / 255.0  # Normalize to [0, 1]
    return images.unsqueeze(1)       # Add channel dimension: [N, 1, 28, 28]


def create_sum_mnist_pairs(images, labels, num_pairs):
    pairs: List[Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]]] = []
    sums = []

    for _ in range(num_pairs):
        i1, i2 = random.randint(0, len(images) - 1), random.randint(0, len(images) - 1)

        img1 = images[i1]
        img2 = images[i2]

        # Concatenate side-by-side: shape becomes [1, 28, 56]
        combined_image = torch.cat((img1, img2), dim=2)

        label_sum = labels[i1] + labels[i2]

        pairs.append(((combined_image.unsqueeze(0), combined_image.unsqueeze(0)), ()))  # Add batch dimension
        sums.append(label_sum)

    # Stack into tensors
    return torch.stack(pairs), torch.tensor(sums)


if __name__ == '__main__':

    # Load the dataset
    train_images = load_images(dataset_path('archive/train-images.idx3-ubyte'))
    train_labels = load_labels(dataset_path('archive/train-labels.idx1-ubyte'))
    train_images = preprocess_images(train_images)

    test_images = load_images(dataset_path('archive/t10k-images.idx3-ubyte'))
    test_labels = load_labels(dataset_path('archive/t10k-labels.idx1-ubyte'))
    test_images = preprocess_images(test_images)

    print(train_images.shape)
    print(train_labels.shape)

    image = train_images[1]
    plt.imshow(image.squeeze(), cmap='gray')
    plt.show()

    ### Create data loaders

    batch_size = 64
    train_dataloader = DataLoader(train_images, batch_size=batch_size)
    test_dataloader = DataLoader(test_images, batch_size=batch_size)
    print(test_images.shape)
    X = next(iter(test_dataloader))
    y = next(iter(test_labels))
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")