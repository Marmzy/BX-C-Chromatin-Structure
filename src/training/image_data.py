#!/usr/bin/env python

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from typing import Tuple


class BXCDataset(Dataset):
    def __init__(
        self,
        X_path: str,
        y_path: str,
        transform: transforms.Compose
    ) -> None:
        """init

        Args:
            X_path (str): Path to image paths file
            y_path (str): Path to labels file
            transform (transforms.Compose): Composition of data transformation techniques to apply
        """

        self.images = pd.read_csv(X_path, header=None)
        self.labels = pd.read_csv(y_path, header=None)[0]
        self.transform = transform

    def __len__(
        self
    ) -> int:
        """Returns number of samples in dataset

        Returns:
            int: Number of samples in dataset
        """

        return len(self.labels)

    def __getitem__(
        self,
        idx: int
    ) -> Tuple[torch.Tensor, np.float, str]:
        """Return sample given index

        Args:
            idx (int): Dataset index number

        Returns:
            Tuple[torch.Tensor, np.float, str]: Image, label and filename tuple
        """

        img = Image.open(self.images.values[idx][0]).convert('RGB')
        img = self.transform(img)

        label = np.float(self.labels[idx])
        file = self.images.iloc[idx, :][0].split("/")[-1].split(".")[0]
        return img, label, file


def get_image_mean(
    target_file: str,
    label_file: str,
    batch: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get mean and std of train dataset

    Args:
        target_file (str): File of dataset image paths
        label_file (str): File of dataset labels
        batch (int): Minibatch size

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Train dataset mean and std values as tensors
    """

    #Defining image transformation techinques to apply
    transformations = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=45),
            transforms.ToTensor()
    ])

    #Creating an image dataset
    bxc_dataset = BXCDataset(target_file, label_file, transformations)
    image_loader = DataLoader(bxc_dataset, batch_size=batch, shuffle=False, pin_memory=True)

    #Looping over the minibatches and calculating the running mean and std
    channels_sum, channels_sqrd_sum, num_batches = 0, 0, 0
    for data, _, _ in image_loader:
        channels_sum += torch.mean(torch.Tensor.float(data), dim=[0, 2, 3])
        channels_sqrd_sum += torch.mean(torch.Tensor.float(data) ** 2, dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches
    std = (channels_sqrd_sum / num_batches - mean ** 2) ** 0.5
    return mean, std


def get_weights(
    file: str,
    verbose: bool
) -> torch.Tensor:
    """Calculate class imbalance weights

    Args:
        file (str): Training fold labels file
        verbose (bool): Detailed output boolean

    Returns:
        torch.Tensor: Class weights tensor
    """

    #Loading the data
    with open(file, "r") as f:
        y_train = [line.rstrip() for line in f]
    y = torch.Tensor([int(l) for l in y_train])

    #Checking for class imbalance
    if verbose:
        print("\nChecking for class imbalance...")
        print("\tNumber of 0 images: {}".format(str(int((y==0.).sum()))))
        print("\tNumber of 1 images: {}".format(str(int(y.sum()))))

    #Calculate the weight for the positive class to alleviate class imbalance
    return (y==0.).sum()/y.sum()