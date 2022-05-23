#!/usr/bin/env python

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms

from PIL import Image
from torch.utils.data import DataLoader, Dataset


class BXCDataset(Dataset):
    def __init__(self, X_path, y_path, transform):
        self.images = pd.read_csv(X_path, header=None)
        self.transform = transform
        print(self.images)

        # quit()

        #Encoding the labels
        labels = pd.read_csv(y_path, header=None)
        self.labels = labels[0]
        # le = preprocessing.LabelEncoder()
        # self.labels = le.fit_transform(labels[0])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = Image.open(self.images.values[idx][0])
        img = img.convert("I")
        img = self.transform(img)

        label = self.labels[idx]
        file = self.images.iloc[idx, :][0].split("/")[-1].split(".")[0]
        return img, label, file


def get_image_mean(target_file, label_file, batch):

    #Defining image transformation techinques to apply
    transformations = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=45),
            transforms.Lambda(lambda img: torch.from_numpy(np.array(img).astype(np.float32)).unsqueeze(0))
    ])

    #Creating an image dataset
    bxc_dataset = BXCDataset(target_file, label_file, transformations)
    # image_loader = DataLoader(bxc_dataset, batch_size=batch, shuffle=False, pin_memory=True)