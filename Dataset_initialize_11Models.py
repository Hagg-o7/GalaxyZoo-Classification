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

class CustomImageDataset_Class1(torch.utils.data.Dataset):
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
        labels = self.img_labels.iloc[idx, 1:4].to_numpy()
        labels = labels.astype('float').reshape(-1, 3)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            labels = self.target_transform(labels)
        sample = {'image': image, 'labels': labels}

        return sample
    
class CustomImageDataset_Class2(torch.utils.data.Dataset):
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
        labels = self.img_labels.iloc[idx, 4:6].to_numpy()
        labels = labels.astype('float').reshape(-1, 2)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            labels = self.target_transform(labels)
        sample = {'image': image, 'labels': labels}

        return sample
    
class CustomImageDataset_Class3(torch.utils.data.Dataset):
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
        labels = self.img_labels.iloc[idx, 6:8].to_numpy()
        labels = labels.astype('float').reshape(-1, 2)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            labels = self.target_transform(labels)
        sample = {'image': image, 'labels': labels}

        return sample
    
class CustomImageDataset_Class4(torch.utils.data.Dataset):
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
        labels = self.img_labels.iloc[idx, 8:10].to_numpy()
        labels = labels.astype('float').reshape(-1, 2)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            labels = self.target_transform(labels)
        sample = {'image': image, 'labels': labels}

        return sample
    
class CustomImageDataset_Class5(torch.utils.data.Dataset):
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
        labels = self.img_labels.iloc[idx, 10:14].to_numpy()
        labels = labels.astype('float').reshape(-1, 4)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            labels = self.target_transform(labels)
        sample = {'image': image, 'labels': labels}

        return sample
    
class CustomImageDataset_Class6(torch.utils.data.Dataset):
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
        labels = self.img_labels.iloc[idx, 14:16].to_numpy()
        labels = labels.astype('float').reshape(-1, 2)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            labels = self.target_transform(labels)
        sample = {'image': image, 'labels': labels}

        return sample
    
class CustomImageDataset_Class7(torch.utils.data.Dataset):
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
        labels = self.img_labels.iloc[idx, 16:19].to_numpy()
        labels = labels.astype('float').reshape(-1, 3)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            labels = self.target_transform(labels)
        sample = {'image': image, 'labels': labels}

        return sample
    
class CustomImageDataset_Class8(torch.utils.data.Dataset):
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
        labels = self.img_labels.iloc[idx, 19:26].to_numpy()
        labels = labels.astype('float').reshape(-1, 7)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            labels = self.target_transform(labels)
        sample = {'image': image, 'labels': labels}

        return sample
    
class CustomImageDataset_Class9(torch.utils.data.Dataset):
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
        labels = self.img_labels.iloc[idx, 26:29].to_numpy()
        labels = labels.astype('float').reshape(-1, 3)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            labels = self.target_transform(labels)
        sample = {'image': image, 'labels': labels}

        return sample
    
class CustomImageDataset_Class10(torch.utils.data.Dataset):
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
        labels = self.img_labels.iloc[idx, 29:32].to_numpy()
        labels = labels.astype('float').reshape(-1, 3)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            labels = self.target_transform(labels)
        sample = {'image': image, 'labels': labels}

        return sample
    
class CustomImageDataset_Class11(torch.utils.data.Dataset):
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
        labels = self.img_labels.iloc[idx, 32:38].to_numpy()
        labels = labels.astype('float').reshape(-1, 6)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            labels = self.target_transform(labels)
        sample = {'image': image, 'labels': labels}

        return sample

data_transform = transforms.Compose([
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.0930, 0.0806, 0.0630], std=[0.1381, 0.1159, 0.1003]),
])


GalaxyZoo_dataset_training_Class1 = CustomImageDataset_Class1("/home/harshit/vscode/git/GalaxyZoo Classification/training_solutions_rev1.csv", "/home/harshit/vscode/git/GalaxyZoo Classification/images_training_rev1", transform = data_transform, target_transform=None)
GalaxyZoo_dataset_training_Class2 = CustomImageDataset_Class2("/home/harshit/vscode/git/GalaxyZoo Classification/training_solutions_rev1.csv", "/home/harshit/vscode/git/GalaxyZoo Classification/images_training_rev1", transform = data_transform, target_transform=None)
GalaxyZoo_dataset_training_Class3 = CustomImageDataset_Class3("/home/harshit/vscode/git/GalaxyZoo Classification/training_solutions_rev1.csv", "/home/harshit/vscode/git/GalaxyZoo Classification/images_training_rev1", transform = data_transform, target_transform=None)
GalaxyZoo_dataset_training_Class4 = CustomImageDataset_Class4("/home/harshit/vscode/git/GalaxyZoo Classification/training_solutions_rev1.csv", "/home/harshit/vscode/git/GalaxyZoo Classification/images_training_rev1", transform = data_transform, target_transform=None)
GalaxyZoo_dataset_training_Class5 = CustomImageDataset_Class5("/home/harshit/vscode/git/GalaxyZoo Classification/training_solutions_rev1.csv", "/home/harshit/vscode/git/GalaxyZoo Classification/images_training_rev1", transform = data_transform, target_transform=None)
GalaxyZoo_dataset_training_Class6 = CustomImageDataset_Class6("/home/harshit/vscode/git/GalaxyZoo Classification/training_solutions_rev1.csv", "/home/harshit/vscode/git/GalaxyZoo Classification/images_training_rev1", transform = data_transform, target_transform=None)
GalaxyZoo_dataset_training_Class7 = CustomImageDataset_Class7("/home/harshit/vscode/git/GalaxyZoo Classification/training_solutions_rev1.csv", "/home/harshit/vscode/git/GalaxyZoo Classification/images_training_rev1", transform = data_transform, target_transform=None)
GalaxyZoo_dataset_training_Class8 = CustomImageDataset_Class8("/home/harshit/vscode/git/GalaxyZoo Classification/training_solutions_rev1.csv", "/home/harshit/vscode/git/GalaxyZoo Classification/images_training_rev1", transform = data_transform, target_transform=None)
GalaxyZoo_dataset_training_Class9 = CustomImageDataset_Class9("/home/harshit/vscode/git/GalaxyZoo Classification/training_solutions_rev1.csv", "/home/harshit/vscode/git/GalaxyZoo Classification/images_training_rev1", transform = data_transform, target_transform=None)
GalaxyZoo_dataset_training_Class10 = CustomImageDataset_Class10("/home/harshit/vscode/git/GalaxyZoo Classification/training_solutions_rev1.csv", "/home/harshit/vscode/git/GalaxyZoo Classification/images_training_rev1", transform = data_transform, target_transform=None)
GalaxyZoo_dataset_training_Class11 = CustomImageDataset_Class11("/home/harshit/vscode/git/GalaxyZoo Classification/training_solutions_rev1.csv", "/home/harshit/vscode/git/GalaxyZoo Classification/images_training_rev1", transform = data_transform, target_transform=None)

GalaxyZoo_dataset_test_Class1 = CustomImageDataset_Class1("/home/harshit/vscode/git/GalaxyZoo Classification/all_zeros_benchmark.csv", "/home/harshit/vscode/git/GalaxyZoo Classification/images_test_rev1", transform=data_transform, target_transform=None)
GalaxyZoo_dataset_test_Class2 = CustomImageDataset_Class2("/home/harshit/vscode/git/GalaxyZoo Classification/all_zeros_benchmark.csv", "/home/harshit/vscode/git/GalaxyZoo Classification/images_test_rev1", transform=data_transform, target_transform=None)
GalaxyZoo_dataset_test_Class3 = CustomImageDataset_Class3("/home/harshit/vscode/git/GalaxyZoo Classification/all_zeros_benchmark.csv", "/home/harshit/vscode/git/GalaxyZoo Classification/images_test_rev1", transform=data_transform, target_transform=None)
GalaxyZoo_dataset_test_Class4 = CustomImageDataset_Class4("/home/harshit/vscode/git/GalaxyZoo Classification/all_zeros_benchmark.csv", "/home/harshit/vscode/git/GalaxyZoo Classification/images_test_rev1", transform=data_transform, target_transform=None)
GalaxyZoo_dataset_test_Class5 = CustomImageDataset_Class5("/home/harshit/vscode/git/GalaxyZoo Classification/all_zeros_benchmark.csv", "/home/harshit/vscode/git/GalaxyZoo Classification/images_test_rev1", transform=data_transform, target_transform=None)
GalaxyZoo_dataset_test_Class6 = CustomImageDataset_Class6("/home/harshit/vscode/git/GalaxyZoo Classification/all_zeros_benchmark.csv", "/home/harshit/vscode/git/GalaxyZoo Classification/images_test_rev1", transform=data_transform, target_transform=None)
GalaxyZoo_dataset_test_Class7 = CustomImageDataset_Class7("/home/harshit/vscode/git/GalaxyZoo Classification/all_zeros_benchmark.csv", "/home/harshit/vscode/git/GalaxyZoo Classification/images_test_rev1", transform=data_transform, target_transform=None)
GalaxyZoo_dataset_test_Class8 = CustomImageDataset_Class8("/home/harshit/vscode/git/GalaxyZoo Classification/all_zeros_benchmark.csv", "/home/harshit/vscode/git/GalaxyZoo Classification/images_test_rev1", transform=data_transform, target_transform=None)
GalaxyZoo_dataset_test_Class9 = CustomImageDataset_Class9("/home/harshit/vscode/git/GalaxyZoo Classification/all_zeros_benchmark.csv", "/home/harshit/vscode/git/GalaxyZoo Classification/images_test_rev1", transform=data_transform, target_transform=None)
GalaxyZoo_dataset_test_Class10 = CustomImageDataset_Class10("/home/harshit/vscode/git/GalaxyZoo Classification/all_zeros_benchmark.csv", "/home/harshit/vscode/git/GalaxyZoo Classification/images_test_rev1", transform=data_transform, target_transform=None)
GalaxyZoo_dataset_test_Class11 = CustomImageDataset_Class11("/home/harshit/vscode/git/GalaxyZoo Classification/all_zeros_benchmark.csv", "/home/harshit/vscode/git/GalaxyZoo Classification/images_test_rev1", transform=data_transform, target_transform=None)

torch.save(GalaxyZoo_dataset_training_Class1, './train1.pt')
torch.save(GalaxyZoo_dataset_training_Class2, './train2.pt')
torch.save(GalaxyZoo_dataset_training_Class3, './train3.pt')
torch.save(GalaxyZoo_dataset_training_Class4, './train4.pt')
torch.save(GalaxyZoo_dataset_training_Class5, './train5.pt')
torch.save(GalaxyZoo_dataset_training_Class6, './train6.pt')
torch.save(GalaxyZoo_dataset_training_Class7, './train7.pt')
torch.save(GalaxyZoo_dataset_training_Class8, './train8.pt')
torch.save(GalaxyZoo_dataset_training_Class9, './train9.pt')
torch.save(GalaxyZoo_dataset_training_Class10, './train10.pt')
torch.save(GalaxyZoo_dataset_training_Class11, './train11.pt')

torch.save(GalaxyZoo_dataset_test_Class1, './test1.pt')
torch.save(GalaxyZoo_dataset_test_Class2, './test2.pt')
torch.save(GalaxyZoo_dataset_test_Class3, './test3.pt')
torch.save(GalaxyZoo_dataset_test_Class4, './test4.pt')
torch.save(GalaxyZoo_dataset_test_Class5, './test5.pt')
torch.save(GalaxyZoo_dataset_test_Class6, './test6.pt')
torch.save(GalaxyZoo_dataset_test_Class7, './test7.pt')
torch.save(GalaxyZoo_dataset_test_Class8, './test8.pt')
torch.save(GalaxyZoo_dataset_test_Class9, './test9.pt')
torch.save(GalaxyZoo_dataset_test_Class10, './test10.pt')
torch.save(GalaxyZoo_dataset_test_Class11, './test11.pt')







