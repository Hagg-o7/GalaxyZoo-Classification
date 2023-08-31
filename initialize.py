import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import pandas as pd
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms


dim_check_img = r"/home/harshit/vscode/git/GalaxyZoo Classification/images_training_rev1/993040.jpg"
test_img_example = cv.imread(dim_check_img)
print(test_img_example.shape)

batch_size = 64
num_classes = 10
learning_rate = 0.001
num_epochs = 20

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


all_transforms = transforms.Compose([transforms.Resize((424,424,3)),
                                     transforms.ToTensor(),
                                     ])
# Create Training dataset
train_dataset = torchvision.datasets(root = '/home/harshit/vscode/git/GalaxyZoo Classification/images_training_rev1',
                                             train = True,
                                             transform = all_transforms)

# Create Testing dataset
test_dataset = torchvision.datasets(root = '/home/harshit/vscode/git/GalaxyZoo Classification/images_test_rev1',
                                            train = False,
                                            transform = all_transforms)

# Instantiate loader objects to facilitate processing
train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                           batch_size = batch_size,
                                           shuffle = True)


test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                           batch_size = batch_size,
                                           shuffle = True)
