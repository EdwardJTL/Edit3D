"""Datasets"""

import os
from typing import List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
import torchvision.transforms as transforms
import torchvision
import glob
import PIL
import random
import math
import pickle
import numpy as np


class CelebA(Dataset):
    """CelelebA Dataset"""

    def __init__(self, dataset_path, img_size, **kwargs):
        super().__init__()

        self.data = glob.glob(dataset_path)
        assert (
            len(self.data) > 0
        ), "Can't find data; make sure you specify the path to your dataset"
        self.transform = transforms.Compose(
            [
                transforms.Resize(320),
                transforms.CenterCrop(256),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Resize((img_size, img_size), interpolation=0),
            ]
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        X = PIL.Image.open(self.data[index])
        X = self.transform(X)

        return X, 0


class Cats(Dataset):
    """Cats Dataset"""

    def __init__(self, dataset_path, img_size, **kwargs):
        super().__init__()

        self.data = glob.glob(dataset_path)
        assert (
            len(self.data) > 0
        ), "Can't find data; make sure you specify the path to your dataset"
        self.transform = transforms.Compose(
            [
                transforms.Resize((img_size, img_size), interpolation=0),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
                transforms.RandomHorizontalFlip(p=0.5),
            ]
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        X = PIL.Image.open(self.data[index])
        X = self.transform(X)

        return X, 0


class Carla(Dataset):
    """Carla Dataset"""

    def __init__(self, dataset_path, img_size, **kwargs):
        super().__init__()

        self.data = glob.glob(dataset_path)
        assert (
            len(self.data) > 0
        ), "Can't find data; make sure you specify the path to your dataset"
        self.transform = transforms.Compose(
            [
                transforms.Resize((img_size, img_size), interpolation=0),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        X = PIL.Image.open(self.data[index])
        X = self.transform(X)

        return X, 0


class ShapeNetCars(Dataset):
    """ShapeNet Cars Dataset"""

    def __init__(self, dataset_path, img_size, **kwargs):
        super().__init__()

        self.data = glob.glob(dataset_path)
        assert (
            len(self.data) > 0
        ), "Can't find data; make sure you specify the path to your dataset"
        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    (img_size, img_size),
                    interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
                ),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        X = PIL.Image.open(self.data[index])
        X = self.transform(X)

        return X, 0


class LSUNCars(Dataset):
    """LSUN Cars Dataset"""

    def __init__(self, dataset_path, img_size, **kwargs):
        super().__init__()

        self.data = glob.glob(dataset_path)
        assert (
            len(self.data) > 0
        ), "Can't find data; make sure you specify the path to your dataset"
        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    img_size,
                    interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
                ),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        X = PIL.Image.open(self.data[index])
        X = self.transform(X)

        return X, 0


class DatasetGANDataset(Dataset):
    def __init__(self, dataset_path, img_size, **kwargs):
        super().__init__()

        self.dataset_path = dataset_path

        self.samples = self.make_dataset(dataset_path)
        if len(self.samples) == 0:
            msg = "Found 0 files in subfolders of: {}\n".format(self.dataset_path)
            raise RuntimeError(msg)

        self.image_transform = transforms.Compose(
            [
                transforms.Resize(
                    img_size,
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        self.map_transform = transforms.Compose(
            [
                transforms.Resize(
                    img_size,
                    interpolation=transforms.InterpolationMode.NEAREST
                ),
                transforms.CenterCrop(img_size)
            ]
        )

        return

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        X = PIL.Image.open(sample[0])
        X = self.image_transform(X)

        map = np.load(sample[1])
        map = torch.from_numpy(map).unsqueeze(dim=0)
        map = self.map_transform(map)

        return torch.cat((X, map)), 0

    @classmethod
    def make_dataset(cls, directory: str) -> List[Tuple[str, str]]:
        instances = []
        root_dir = os.path.abspath(directory)
        if not os.path.isdir(root_dir):
            raise ValueError("Invalid input directory")
        for root, _, fnames in sorted(os.walk(root_dir, followlinks=False)):
            tmp = {}
            for f in fnames:
                if f.lower().endswith(('png', 'npy')) and (not 'latent' in f):
                    pathname, extension = os.path.splitext(f)
                    num = pathname.split('_')[-1]
                    tmp[num] = tmp.get(num, []) + [os.path.join(root, f)]
            instances += list(map(lambda x: sorted(x), filter(lambda x: len(x) == 2, tmp.values())))
        return instances


def get_dataset(name, subsample=None, batch_size=1, **kwargs):
    dataset = globals()[name](**kwargs)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=False,
        num_workers=8,
    )
    return dataloader, 3


def get_dataset_distributed(name, world_size, rank, batch_size, **kwargs):
    dataset = globals()[name](**kwargs)

    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        pin_memory=True,
        num_workers=4,
    )

    return dataloader, 3
