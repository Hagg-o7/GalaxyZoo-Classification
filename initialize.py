import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


dim_check_img = r"/home/harshit/vscode/git/GalaxyZoo Classification/images_training_rev1/993040.jpg"
test_img_example = cv.imread(dim_check_img)
print(test_img_example.shape)

