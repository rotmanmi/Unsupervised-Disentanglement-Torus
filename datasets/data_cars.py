# Adapted from https://github.com/genyrosk/pytorch-VAE-models/

from __future__ import print_function
import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms
import scipy.io as sio
import PIL
import os

"""
Adapted from  https://github.com/google-research/disentanglement_lib/blob/master/disentanglement_lib/data/ground_truth/cars3d.py

"""


def _load_mesh(filename):
    """Parses a single source file and rescales contained images."""
    with open(filename, "rb") as f:
        mesh = np.einsum("abcde->deabc", sio.loadmat(f)["im"])
    flattened_mesh = mesh.reshape((-1,) + mesh.shape[2:])
    rescaled_mesh = np.zeros((flattened_mesh.shape[0], 64, 64, 3))
    for i in range(flattened_mesh.shape[0]):
        pic = PIL.Image.fromarray(flattened_mesh[i, :, :, :])
        pic.thumbnail((64, 64), PIL.Image.ANTIALIAS)
        rescaled_mesh[i, :, :, :] = np.array(pic)
    return rescaled_mesh * 1. / 255


def _load_data(path):
    all_files = [x for x in os.listdir(os.path.join(path, 'cars')) if ".mat" in x]
    dataset = []
    labels = []
    for i, filename in enumerate(all_files):
        data_mesh = _load_mesh(os.path.join(path, 'cars', filename))
        factor1 = np.array(list(range(4)))
        factor2 = np.array(list(range(24)))
        all_factors = np.transpose([
            np.tile(factor1, len(factor2)),
            np.repeat(factor2, len(factor1)),
            np.tile(i,
                    len(factor1) * len(factor2))
        ])
        # indexes = index.features_to_index(all_factors)
        labels.append(all_factors)
        dataset.append(data_mesh)
    return np.concatenate(dataset, 0), np.concatenate(labels, 0)


# Ignore warnings
import warnings

warnings.filterwarnings("ignore")


class Disentangled3DCarsDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, dir, transform=None):
        """
        Args:
            dir (string): Directory containing the dSprites dataset
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.dir = dir

        imgs, self.latents_values = _load_data(dir)

        self.imgs = (2 * imgs) - 1

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


def load_3dcars(root, val_split=0.8, shuffle=True, seed=42):
    # img_size = 64
    path = os.path.join(root, '3dcars')
    dataset = Disentangled3DCarsDataset(path, transform=transforms.ToTensor())

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


