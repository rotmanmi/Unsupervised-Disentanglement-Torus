# Adapted from https://github.com/genyrosk/pytorch-VAE-models/

from __future__ import print_function
import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Ignore warnings
import warnings

warnings.filterwarnings("ignore")


class Disentangled2DShapesDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, dir, transform=None):
        """
        Args:
            dir (string): Directory containing the dSprites dataset
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.dir = dir
        self.filename = 'shapes.npz'
        self.filepath = f'{self.dir}/{self.filename}'
        dataset_zip = np.load(self.filepath, allow_pickle=True, encoding='bytes')

        # print('Keys in the dataset:', dataset_zip.keys())
        self.imgs = 2 * (dataset_zip['images'] / 255.0) - 1
        self.latents_values = dataset_zip['gts']

        # print('Metadata: \n', self.metadata)
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        sample = self.imgs[idx].astype(np.float32)
        # sample = sample.reshape(1, sample.shape[0], sample.shape[1])
        if self.transform:
            sample = self.transform(sample)
        return sample, self.latents_values[idx]


def load_2dshapes(root, val_split=0.8, shuffle=True, seed=42):
    # img_size = 64
    path = os.path.join(root, 'shapes')
    dataset = Disentangled2DShapesDataset(path, transform=transforms.ToTensor())

    # Create data indices for training and validation splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(val_split * dataset_size))
    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[:split], indices[split:]

    # Create data samplers and loaders:
    train_sampler = Subset(dataset, train_indices)
    val_sampler = Subset(dataset, val_indices)

    return train_sampler, val_sampler

