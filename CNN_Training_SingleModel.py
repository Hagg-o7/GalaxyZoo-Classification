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

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

from Dataset_Initialize_SingleModel import CustomImageDataset_SingleModel

train_SingleModel = torch.load('./train_SingleModel.pt')

train_set_SingleModel, val_set_SingleModel = torch.utils.data.random_split(train_SingleModel, [int(0.8*len(train_SingleModel)), len(train_SingleModel) - int(0.8*len(train_SingleModel))])

train_loader_SingleModel = DataLoader(train_SingleModel, batch_size=32, shuffle=True)

val_loader_SingleModel = DataLoader(val_set_SingleModel, batch_size=32, shuffle=True)


# Define AlexNet Architecture
class AlexNet(nn.Module):
    def __init__(self, num_classes):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x

# Initialize model
model_SingleModel = AlexNet(num_classes=37)

# Loss and Optimizer
criterion = nn.BCEWithLogitsLoss()  # Suitable for multi-label classification
optimizer = optim.Adam(model_SingleModel.parameters(), lr=0.001)

# Training Loop

#Model
num_epochs = 10
best_val_loss = float('inf')
for epoch in range(num_epochs):
    model_SingleModel.train()
    for i, data in enumerate(train_loader_SingleModel):
        images = data['image']
        labels = data['labels'] 
        # Forward pass
        outputs = model_SingleModel(images)
        loss = criterion(outputs, labels.float().squeeze(1))

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader_SingleModel)}], Loss: {loss.item()}')

    # Validation
    model_SingleModel.eval()
    total_val_loss = 0
    with torch.no_grad():
        for images, labels in val_loader_SingleModel:
            outputs = model_SingleModel(images)
            val_loss = criterion(outputs, labels.float())
            total_val_loss += val_loss.item()
    
    average_val_loss = total_val_loss / len(val_loader_SingleModel)
    print(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {average_val_loss}')

    # Save the best model
    if average_val_loss < best_val_loss:
        best_val_loss = average_val_loss
        torch.save(model_SingleModel.state_dict(), 'best_model_SingleModel.pth')
