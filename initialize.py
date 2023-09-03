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


dim_check_img = r"/home/harshit/vscode/git/GalaxyZoo Classification/images_training_rev1/993040.jpg"
test_img_example = cv.imread(dim_check_img)
print(test_img_example.shape)

batch_size = 64
num_classes = 10
learning_rate = 0.001
num_epochs = 20

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


import os
import pandas as pd
from torchvision.io import read_image

class CustomImageDataset:
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

GalaxyZoo_dataset_training = CustomImageDataset("/home/harshit/vscode/git/GalaxyZoo Classification/training_solutions_rev1.csv", "/home/harshit/vscode/git/GalaxyZoo Classification/images_training_rev1", None, None)
GalaxyZoo_dataset_test = CustomImageDataset("/home/harshit/vscode/git/GalaxyZoo Classification/all_zeros_benchmark.csv", "/home/harshit/vscode/git/GalaxyZoo Classification/images_test_rev1", None, None)

from torch.utils.data import DataLoader

train_dataloader = DataLoader(GalaxyZoo_dataset_training, batch_size=64, shuffle=True)
test_dataloader = DataLoader(GalaxyZoo_dataset_test, batch_size=64, shuffle=True)


