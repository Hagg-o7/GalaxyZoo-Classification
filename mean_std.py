import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import pandas as pd
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import os
from torchvision.io import read_image
from tqdm import tqdm
from torch.utils.data import DataLoader
from PIL import Image

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class CustomImageDataset(torch.utils.data.Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_name = str(self.img_labels.iloc[idx, 0]) + '.jpg'  # Append the file extension
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path)
        labels = self.img_labels.iloc[idx, 1:].to_numpy()
        labels = labels.astype('float').reshape(-1, 37)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            labels = self.target_transform(labels)
        sample = {'image': image, 'labels': labels}

        return sample

data_transform = transforms.Compose([
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
])

GalaxyZoo_dataset_training = CustomImageDataset("/home/harshit/vscode/git/GalaxyZoo Classification/training_solutions_rev1.csv", "/home/harshit/vscode/git/GalaxyZoo Classification/images_training_rev1", transform = data_transform, target_transform=None)


# DataLoader
train_dataloader = DataLoader(GalaxyZoo_dataset_training, batch_size=64, shuffle=True)

# Function to calculate mean and std
def calculate_mean_std(train_dataloader):
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0

    for batch in tqdm(train_dataloader):
        images = batch['image']  # Access the 'image' key correctly
        channels_sum += torch.mean(images, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(images**2, dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches
    std = (channels_squared_sum / num_batches - mean**2)**0.5

    return mean, std

# Calculate mean and std
mean, std = calculate_mean_std(train_dataloader)
print(f"Mean: {mean}")
print(f"Std: {std}")