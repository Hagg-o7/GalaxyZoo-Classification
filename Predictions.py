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

from CNN_Training_11Models import AlexNet

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

model1.load_state_dict(torch.load('best_model1.pth'))
model2.load_state_dict(torch.load('best_model2.pth'))
model3.load_state_dict(torch.load('best_model3.pth'))
model4.load_state_dict(torch.load('best_model4.pth'))
model5.load_state_dict(torch.load('best_model5.pth'))
model6.load_state_dict(torch.load('best_model6.pth'))
model7.load_state_dict(torch.load('best_model7.pth'))
model8.load_state_dict(torch.load('best_model8.pth'))
model9.load_state_dict(torch.load('best_model9.pth'))
model10.load_state_dict(torch.load('best_model10.pth'))
model11.load_state_dict(torch.load('best_model11.pth'))

model1.eval()
model2.eval()
model3.eval()
model4.eval()
model5.eval()
model6.eval()
model7.eval()
model8.eval()
model9.eval()
model10.eval()
model11.eval()

data_transform = transforms.Compose([
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.0930, 0.0806, 0.0630], std=[0.1381, 0.1159, 0.1003]),
])

from Dataset_initialize_11Models import CustomImageDataset_Class1, CustomImageDataset_Class2, CustomImageDataset_Class3, CustomImageDataset_Class4, CustomImageDataset_Class5, CustomImageDataset_Class6, CustomImageDataset_Class7, CustomImageDataset_Class8, CustomImageDataset_Class9, CustomImageDataset_Class10, CustomImageDataset_Class11

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

test_loader1 = DataLoader(test1, batch_size=32, shuffle=True)
test_loader2 = DataLoader(test2, batch_size=32, shuffle=True)
test_loader3 = DataLoader(test3, batch_size=32, shuffle=True)
test_loader4 = DataLoader(test4, batch_size=32, shuffle=True)
test_loader5 = DataLoader(test5, batch_size=32, shuffle=True)
test_loader6 = DataLoader(test6, batch_size=32, shuffle=True)
test_loader7 = DataLoader(test7, batch_size=32, shuffle=True)
test_loader8 = DataLoader(test8, batch_size=32, shuffle=True)
test_loader9 = DataLoader(test9, batch_size=32, shuffle=True)
test_loader10 = DataLoader(test10, batch_size=32, shuffle=True)
test_loader11 = DataLoader(test11, batch_size=32, shuffle=True)


benchmark_labels = pd.read_csv('./all_zeros_benchmark.csv')
predicted_labels = np.zeros(len(benchmark_labels),1)
for i in range(0, len(benchmark_labels)-1):
    predicted_labels[i,0] = benchmark_labels.iloc[i+1, 0]

with torch.no_grad():
    for data, _ in test_loader1:
        images = data['image']
        labels = data['labels']
        outputs = model1(images)
        probabilities = F.softmax(outputs, dim=1)
        predicted_labels = np.hstack(predicted_labels, probabilities.tolist())
    for data, _ in test_loader2:
        images = data['image']
        labels = data['labels']
        outputs = model2(images)
        probabilities = F.softmax(outputs, dim=1)
        predicted_labels = np.hstack(predicted_labels, probabilities.tolist())
    for data, _ in test_loader3:
        images = data['image']
        labels = data['labels']
        outputs = model3(images)
        probabilities = F.softmax(outputs, dim=1)
        predicted_labels = np.hstack(predicted_labels, probabilities.tolist())
    for data, _ in test_loader4:
        images = data['image']
        labels = data['labels']
        outputs = model4(images)
        probabilities = F.softmax(outputs, dim=1)
        predicted_labels = np.hstack(predicted_labels, probabilities.tolist())
    for data, _ in test_loader5:
        images = data['image']
        labels = data['labels']
        outputs = model5(images)
        probabilities = F.softmax(outputs, dim=1)
        predicted_labels = np.hstack(predicted_labels, probabilities.tolist())
    for data, _ in test_loader6:
        images = data['image']
        labels = data['labels']
        outputs = model6(images)
        probabilities = F.softmax(outputs, dim=1)
        predicted_labels = np.hstack(predicted_labels, probabilities.tolist())
    for data, _ in test_loader7:
        images = data['image']
        labels = data['labels']
        outputs = model7(images)
        probabilities = F.softmax(outputs, dim=1)
        predicted_labels = np.hstack(predicted_labels, probabilities.tolist())
    for data, _ in test_loader8:
        images = data['image']
        labels = data['labels']
        outputs = model8(images)
        probabilities = F.softmax(outputs, dim=1)
        predicted_labels = np.hstack(predicted_labels, probabilities.tolist())
    for data, _ in test_loader9:
        images = data['image']
        labels = data['labels']
        outputs = model9(images)
        probabilities = F.softmax(outputs, dim=1)
        predicted_labels = np.hstack(predicted_labels, probabilities.tolist())
    for data, _ in test_loader10:
        images = data['image']
        labels = data['labels']
        outputs = model10(images)
        probabilities = F.softmax(outputs, dim=1)
        predicted_labels = np.hstack(predicted_labels, probabilities.tolist())
    for data, _ in test_loader11:
        images = data['image']
        labels = data['labels']
        outputs = model11(images)
        probabilities = F.softmax(outputs, dim=1)
        predicted_labels = np.hstack(predicted_labels, probabilities.tolist())



# Create a DataFrame for the predicted probabilities
result_dataframe = pd.DataFrame(predicted_labels, columns=['GalaxyID', 'Class1.1', 'Class1.2', 'Class1.3', 'Class2.1', 'Class2.2', 'Class3.1', 'Class3.2', 'Class4.1', 'Class4.2', 'Class5.1', 'Class5.2', 'Class5.3', 'Class5.4', 'Class6.1', 'Class6.2', 'Class7.1', 'Class7.2', 'Class7.3', 'Class8.1', 'Class8.2', 'Class8.3', 'Class8.4', 'Class8.5', 'Class8.6', 'Class8.7', 'Class9.1', 'Class9.2', 'Class9.3', 'Class10.1', 'Class10.2', 'Class10.3', 'Class11.1', 'Class11.2', 'Class11.3', 'Class11.4', 'Class11.5', 'Class11.6'])
result_dataframe.to_csv('result.csv', index=False)




