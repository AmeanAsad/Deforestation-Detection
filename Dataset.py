# -*- coding: utf-8 -*-
"""
Created on Fri May  6 21:13:47 2022

@author: Amean
"""
from PIL import Image
import torch
from torch.utils.data.dataset import Dataset


class CustomDataset(Dataset):
    """Dataset wrapping images and target labels for Kaggle - Planet Amazon from Space competition.

    Arguments:
        csvPath: A CSV file path
        imgPath: Path to image folder
        imgExtension: Extension of images
        transform: PIL transforms
    """

    def __init__(self, dataFrame, imgPath, imgExtension=".jpg", transform=None):
        
        self.imgPath = imgPath
        self.imgExtension = imgExtension
        self.transform = transform

        self.imageNames = dataFrame["image_name"]
        self.imageLabels = dataFrame["Binary Tag"]

    def __getitem__(self, index):
        img = Image.open( str(self.imgPath/self.imageNames[index]) + self.imgExtension)
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        
        label = torch.tensor(self.imageLabels[index], dtype=torch.long)
        return img, label

    def __len__(self):
        return len(self.imageNames.index)