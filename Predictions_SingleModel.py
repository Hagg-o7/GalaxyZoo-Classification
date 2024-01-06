import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import pandas as pd
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import os
import torch.optim as optim
from torchvision.io import read_image
from tqdm import tqdm
from torch.utils.data import DataLoader
from PIL import Image
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from CNN_Training_SingleModel import AlexNet

model_SingleModel = AlexNet(num_classes=37)


model_SingleModel.load_state_dict(torch.load('best_model_SingleModel.pth'))

model_SingleModel.eval()


from Dataset_Initialize_SingleModel import CustomImageDataset_SingleModel

test_SingleModel = torch.load('./test_SingleModel.pt')

test_loader_SingleModel = DataLoader(test_SingleModel, batch_size=32, shuffle=True)



benchmark_labels = pd.read_csv('./all_zeros_benchmark.csv')
predicted_labels = np.zeros(len(benchmark_labels),1)
for i in range(0, len(benchmark_labels)-1):
    predicted_labels[i,0] = benchmark_labels.iloc[i+1, 0]

with torch.no_grad():
    for data, _ in test_loader_SingleModel:
        images = data['image']
        labels = data['labels']
        outputs = model_SingleModel(images)
        probabilities = F.softmax(outputs, dim=1)
        predicted_labels = np.hstack(predicted_labels, probabilities.tolist())

# Create a DataFrame for the predicted probabilities
result_dataframe = pd.DataFrame(predicted_labels, columns=['GalaxyID', 'Class1.1', 'Class1.2', 'Class1.3', 'Class2.1', 'Class2.2', 'Class3.1', 'Class3.2', 'Class4.1', 'Class4.2', 'Class5.1', 'Class5.2', 'Class5.3', 'Class5.4', 'Class6.1', 'Class6.2', 'Class7.1', 'Class7.2', 'Class7.3', 'Class8.1', 'Class8.2', 'Class8.3', 'Class8.4', 'Class8.5', 'Class8.6', 'Class8.7', 'Class9.1', 'Class9.2', 'Class9.3', 'Class10.1', 'Class10.2', 'Class10.3', 'Class11.1', 'Class11.2', 'Class11.3', 'Class11.4', 'Class11.5', 'Class11.6'])
result_dataframe.to_csv('result_SingleModel.csv', index=False)