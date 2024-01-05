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

from Dataset_initialize import CustomImageDataset_Class1, CustomImageDataset_Class2, CustomImageDataset_Class3, CustomImageDataset_Class4, CustomImageDataset_Class5, CustomImageDataset_Class6, CustomImageDataset_Class7, CustomImageDataset_Class8, CustomImageDataset_Class9, CustomImageDataset_Class10, CustomImageDataset_Class11

train1 = torch.load('./train1.pt')
train2 = torch.load('./train2.pt')
train3 = torch.load('./train3.pt')
train4 = torch.load('./train4.pt')
train5 = torch.load('./train5.pt')
train6 = torch.load('./train6.pt')
train7 = torch.load('./train7.pt')
train8 = torch.load('./train8.pt')
train9 = torch.load('./train9.pt')
train10 = torch.load('./train10.pt')
train11 = torch.load('./train11.pt')

test1 = torch.load('./test1.pt')
test2 = torch.load('./test2.pt')
test3 = torch.load('./test3.pt')
test4 = torch.load('./test4.pt')
test5 = torch.load('./test5.pt')
test6 = torch.load('./test6.pt')
test7 = torch.load('./test7.pt')
test8 = torch.load('./test8.pt')
test9 = torch.load('./test9.pt')
test10 = torch.load('./test10.pt')
test11 = torch.load('./test11.pt')

train_loader1 = DataLoader(train1, batch_size=32, shuffle=True)
train_loader2 = DataLoader(train2, batch_size=32, shuffle=True)
train_loader3 = DataLoader(train3, batch_size=32, shuffle=True)
train_loader4 = DataLoader(train4, batch_size=32, shuffle=True)
train_loader5 = DataLoader(train5, batch_size=32, shuffle=True)
train_loader6 = DataLoader(train6, batch_size=32, shuffle=True)
train_loader7 = DataLoader(train7, batch_size=32, shuffle=True)
train_loader8 = DataLoader(train8, batch_size=32, shuffle=True)
train_loader9 = DataLoader(train9, batch_size=32, shuffle=True)
train_loader10 = DataLoader(train10, batch_size=32, shuffle=True)
train_loader11 = DataLoader(train11, batch_size=32, shuffle=True)

val_loader1 = DataLoader(test1, batch_size=32, shuffle=True)
val_loader2 = DataLoader(test2, batch_size=32, shuffle=True)
val_loader3 = DataLoader(test3, batch_size=32, shuffle=True)
val_loader4 = DataLoader(test4, batch_size=32, shuffle=True)
val_loader5 = DataLoader(test5, batch_size=32, shuffle=True)
val_loader6 = DataLoader(test6, batch_size=32, shuffle=True)
val_loader7 = DataLoader(test7, batch_size=32, shuffle=True)
val_loader8 = DataLoader(test8, batch_size=32, shuffle=True)
val_loader9 = DataLoader(test9, batch_size=32, shuffle=True)
val_loader10 = DataLoader(test10, batch_size=32, shuffle=True)
val_loader11 = DataLoader(test11, batch_size=32, shuffle=True)

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
model1 = AlexNet(num_classes=3)
model2 = AlexNet(num_classes=2)
model3 = AlexNet(num_classes=2)
model4 = AlexNet(num_classes=2)
model5 = AlexNet(num_classes=4)
model6 = AlexNet(num_classes=2)
model7 = AlexNet(num_classes=3)
model8 = AlexNet(num_classes=7)
model9 = AlexNet(num_classes=3)
model10 = AlexNet(num_classes=3)
model11 = AlexNet(num_classes=6)

# Loss and Optimizer
criterion = nn.BCEWithLogitsLoss()  # Suitable for multi-label classification
optimizer = optim.Adam(model1.parameters(), lr=0.001)

# Training Loop

#Model 1
num_epochs = 10
best_val_loss = float('inf')
for epoch in range(num_epochs):
    model1.train()
    for i, data in enumerate(train_loader1):
        images = data['image']
        labels = data['labels'] 
        # Forward pass
        outputs = model1(images)
        loss = criterion(outputs, labels.float().squeeze(1))

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader1)}], Loss: {loss.item()}')

    # Validation
    model1.eval()
    total_val_loss = 0
    with torch.no_grad():
        for images, labels in val_loader1:
            outputs = model1(images)
            val_loss = criterion(outputs, labels.float())
            total_val_loss += val_loss.item()
    
    average_val_loss = total_val_loss / len(val_loader1)
    print(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {average_val_loss}')

    # Save the best model
    if average_val_loss < best_val_loss:
        best_val_loss = average_val_loss
        torch.save(model1.state_dict(), 'best_model1.pth')

#Model 2
num_epochs = 10
best_val_loss = float('inf')
for epoch in range(num_epochs):
    model2.train()
    for i, (images, labels) in enumerate(train_loader2):
        images = data['image']
        labels = data['labels'] 
        # Forward pass
        outputs = model2(images)
        loss = criterion(outputs, labels.float().squeeze(1))

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader2)}], Loss: {loss.item()}')

    # Validation
    model2.eval()
    total_val_loss = 0
    with torch.no_grad():
        for images, labels in val_loader2:
            outputs = model2(images)
            val_loss = criterion(outputs, labels.float())
            total_val_loss += val_loss.item()
    
    average_val_loss = total_val_loss / len(val_loader2)
    print(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {average_val_loss}')

    # Save the best model
    if average_val_loss < best_val_loss:
        best_val_loss = average_val_loss
        torch.save(model2.state_dict(), 'best_model2.pth')

#Model 3
num_epochs = 10
best_val_loss = float('inf')
for epoch in range(num_epochs):
    model3.train()
    for i, (images, labels) in enumerate(train_loader3):
        images = data['image']
        labels = data['labels'] 
        # Forward pass
        outputs = model3(images)
        loss = criterion(outputs, labels.float().squeeze(1))

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader3)}], Loss: {loss.item()}')

    # Validation
    model3.eval()
    total_val_loss = 0
    with torch.no_grad():
        for images, labels in val_loader3:
            outputs = model3(images)
            val_loss = criterion(outputs, labels.float())
            total_val_loss += val_loss.item()
    
    average_val_loss = total_val_loss / len(val_loader3)
    print(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {average_val_loss}')

    # Save the best model
    if average_val_loss < best_val_loss:
        best_val_loss = average_val_loss
        torch.save(model3.state_dict(), 'best_model3.pth')

#Model 4
num_epochs = 10
best_val_loss = float('inf')
for epoch in range(num_epochs):
    model4.train()
    for i, (images, labels) in enumerate(train_loader4):
        images = data['image']
        labels = data['labels'] 
        # Forward pass
        outputs = model4(images)
        loss = criterion(outputs, labels.float().squeeze(1))

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader4)}], Loss: {loss.item()}')

    # Validation
    model4.eval()
    total_val_loss = 0
    with torch.no_grad():
        for images, labels in val_loader4:
            outputs = model4(images)
            val_loss = criterion(outputs, labels.float())
            total_val_loss += val_loss.item()
    
    average_val_loss = total_val_loss / len(val_loader4)
    print(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {average_val_loss}')

    # Save the best model
    if average_val_loss < best_val_loss:
        best_val_loss = average_val_loss
        torch.save(model4.state_dict(), 'best_model4.pth')

#Model 5
num_epochs = 10
best_val_loss = float('inf')
for epoch in range(num_epochs):
    model5.train()
    for i, (images, labels) in enumerate(train_loader5):
        images = data['image']
        labels = data['labels'] 
        # Forward pass
        outputs = model5(images)
        loss = criterion(outputs, labels.float().squeeze(1))

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader5)}], Loss: {loss.item()}')

    # Validation
    model5.eval()
    total_val_loss = 0
    with torch.no_grad():
        for images, labels in val_loader5:
            outputs = model5(images)
            val_loss = criterion(outputs, labels.float())
            total_val_loss += val_loss.item()
    
    average_val_loss = total_val_loss / len(val_loader5)
    print(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {average_val_loss}')

    # Save the best model
    if average_val_loss < best_val_loss:
        best_val_loss = average_val_loss
        torch.save(model5.state_dict(), 'best_model5.pth')

#Model 6
num_epochs = 10
best_val_loss = float('inf')
for epoch in range(num_epochs):
    model6.train()
    for i, (images, labels) in enumerate(train_loader6):
        images = data['image']
        labels = data['labels'] 
        # Forward pass
        outputs = model6(images)
        loss = criterion(outputs, labels.float().squeeze(1))

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader6)}], Loss: {loss.item()}')

    # Validation
    model6.eval()
    total_val_loss = 0
    with torch.no_grad():
        for images, labels in val_loader6:
            outputs = model6(images)
            val_loss = criterion(outputs, labels.float())
            total_val_loss += val_loss.item()
    
    average_val_loss = total_val_loss / len(val_loader6)
    print(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {average_val_loss}')

    # Save the best model
    if average_val_loss < best_val_loss:
        best_val_loss = average_val_loss
        torch.save(model6.state_dict(), 'best_model6.pth')

#Model 7
num_epochs = 10
best_val_loss = float('inf')
for epoch in range(num_epochs):
    model7.train()
    for i, (images, labels) in enumerate(train_loader7):
        images = data['image']
        labels = data['labels'] 
        # Forward pass
        outputs = model7(images)
        loss = criterion(outputs, labels.float().squeeze(1))

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader7)}], Loss: {loss.item()}')

    # Validation
    model7.eval()
    total_val_loss = 0
    with torch.no_grad():
        for images, labels in val_loader7:
            outputs = model7(images)
            val_loss = criterion(outputs, labels.float())
            total_val_loss += val_loss.item()
    
    average_val_loss = total_val_loss / len(val_loader7)
    print(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {average_val_loss}')

    # Save the best model
    if average_val_loss < best_val_loss:
        best_val_loss = average_val_loss
        torch.save(model7.state_dict(), 'best_model7.pth')

#Model 8
num_epochs = 10
best_val_loss = float('inf')
for epoch in range(num_epochs):
    model8.train()
    for i, (images, labels) in enumerate(train_loader8):
        images = data['image']
        labels = data['labels'] 
        # Forward pass
        outputs = model8(images)
        loss = criterion(outputs, labels.float().squeeze(1))

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader8)}], Loss: {loss.item()}')

    # Validation
    model8.eval()
    total_val_loss = 0
    with torch.no_grad():
        for images, labels in val_loader8:
            outputs = model8(images)
            val_loss = criterion(outputs, labels.float())
            total_val_loss += val_loss.item()
    
    average_val_loss = total_val_loss / len(val_loader8)
    print(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {average_val_loss}')

    # Save the best model
    if average_val_loss < best_val_loss:
        best_val_loss = average_val_loss
        torch.save(model8.state_dict(), 'best_model8.pth')

#Model 9
num_epochs = 10
best_val_loss = float('inf')
for epoch in range(num_epochs):
    model9.train()
    for i, (images, labels) in enumerate(train_loader9):
        images = data['image']
        labels = data['labels'] 
        # Forward pass
        outputs = model9(images)
        loss = criterion(outputs, labels.float().squeeze(1))

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader9)}], Loss: {loss.item()}')

    # Validation
    model9.eval()
    total_val_loss = 0
    with torch.no_grad():
        for images, labels in val_loader9:
            outputs = model9(images)
            val_loss = criterion(outputs, labels.float())
            total_val_loss += val_loss.item()
    
    average_val_loss = total_val_loss / len(val_loader9)
    print(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {average_val_loss}')

    # Save the best model
    if average_val_loss < best_val_loss:
        best_val_loss = average_val_loss
        torch.save(model9.state_dict(), 'best_model9.pth')

#Model 10
num_epochs = 10
best_val_loss = float('inf')
for epoch in range(num_epochs):
    model10.train()
    for i, (images, labels) in enumerate(train_loader10):
        images = data['image']
        labels = data['labels'] 
        # Forward pass
        outputs = model10(images)
        loss = criterion(outputs, labels.float().squeeze(1))

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader10)}], Loss: {loss.item()}')

    # Validation
    model10.eval()
    total_val_loss = 0
    with torch.no_grad():
        for images, labels in val_loader10:
            outputs = model10(images)
            val_loss = criterion(outputs, labels.float())
            total_val_loss += val_loss.item()
    
    average_val_loss = total_val_loss / len(val_loader10)
    print(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {average_val_loss}')

    # Save the best model
    if average_val_loss < best_val_loss:
        best_val_loss = average_val_loss
        torch.save(model10.state_dict(), 'best_model10.pth')

#Model 11
num_epochs = 10
best_val_loss = float('inf')
for epoch in range(num_epochs):
    model11.train()
    for i, (images, labels) in enumerate(train_loader11):
        images = data['image']
        labels = data['labels'] 
        # Forward pass
        outputs = model11(images)
        loss = criterion(outputs, labels.float().squeeze(1))

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader11)}], Loss: {loss.item()}')

    # Validation
    model11.eval()
    total_val_loss = 0
    with torch.no_grad():
        for images, labels in val_loader11:
            outputs = model11(images)
            val_loss = criterion(outputs, labels.float())
            total_val_loss += val_loss.item()
    
    average_val_loss = total_val_loss / len(val_loader11)
    print(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {average_val_loss}')

    # Save the best model
    if average_val_loss < best_val_loss:
        best_val_loss = average_val_loss
        torch.save(model11.state_dict(), 'best_model11.pth')








