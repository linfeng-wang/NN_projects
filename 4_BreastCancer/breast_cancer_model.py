
#%%
from array import array
from cmath import nan
from pyexpat import model
import statistics
from tkinter.ttk import Separator
import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchviz import make_dot
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.utils.data.dataset import random_split
from torchvision import datasets, transforms
from torchvision.io import read_image
from torch.autograd import variable

from itertools import chain
from sklearn import metrics as met
import pickle
from icecream import ic
import re

import matplotlib.pyplot as plt
import pathlib
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from importlib import reload
# import util
# import model_torch_simple
# from torchmetrics import Accuracy
from tqdm import tqdm
import argparse

import numpy as np
from PIL import Image
import time
import os
import copy
import glob
import random

# %%
device = 'cuda' if torch.cuda.is_available() else 'cpu'

all_samples = glob.glob('data/archive/*')

# def imshow(inp, title): # inp is a tensor
#     """Imshow for Tensor."""
#     inp = inp.numpy().transpose((1, 2, 0)) # convert to numpy array
#     inp = std * inp + mean # unnormalize
#     inp = np.clip(inp, 0, 1) # clip to 0-1
#     plt.imshow(inp)
#     plt.title(title)
#     plt.show()

import os
import glob

#%%
# Specify the directory path
directory_path = 'data/archive'
# Define an empty list to store the file paths
file_paths = []

# Walk through the directory tree and add the path of each file to the list
for dirpath, dirnames, filenames in os.walk(directory_path):
    for filename in filenames:
        file_path = os.path.join(dirpath, filename)
        if os.path.isfile(file_path):  # Check if it's a file and not a directory
            file_paths.append(file_path)

# Print the list of file paths
# Print the list of file paths
# print(file_paths)

#read_image('/Users/linfengwang/Github/NN_projects/4_BreastCancer/data/archive/8863/0/8863_idx5_x51_y1251_class0.png')
#%%
file_paths = [file for file in file_paths if not file.endswith('.DS_Store')]
#%%
random.seed(123)
random.shuffle(file_paths) # shuffle the file paths so that 0 and 1 are mixed
#%%
def extract_class_number(input_string):    
    match = re.search(r'class(\d+)\.png', input_string)
# Extract the class number from the match object
    if match:
        return int(match.group(1))
    else:
        pass

class_number = [extract_class_number(x) for x in file_paths]
# class_number = class_number[2:]
# print(extract_class_number('/Users/linfengwang/Github/NN_projects/4_BreastCancer/data/archive/8863/0/8863_idx5_x51_y1251_class192038920.png')) # testing

#%%
torch_read =read_image('/Users/linfengwang/Github/NN_projects/4_BreastCancer/data/archive/8863/0/8863_idx5_x51_y1251_class0.png')
array_read = np.array(Image.open('/Users/linfengwang/Github/NN_projects/4_BreastCancer/data/archive/8863/0/8863_idx5_x51_y1251_class0.png'))

transform1 = transforms.Compose([
            transforms.Resize((50, 50)),
            transforms.ToTensor()
        ])

#%%
class breast_cancer_dataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.transform = transforms.Compose([
            transforms.Resize((50, 50)),
        ])
    def __getitem__(self, index):
        img = read_image(self.x[index])
        # img = np.array(Image.open(self.x[index]))
        img = self.transform(img)
        class_type = self.y[index]
        return img, class_type
    def __len__(self):
        return len(self.x)

training_dataset = breast_cancer_dataset(file_paths, class_number)

train_dataset, val_dataset = random_split(training_dataset, [int(len(training_dataset)*0.8), len(training_dataset)-int(len(training_dataset)*0.8)])

#%%
epoch = 20
batch_size = 128
lr = 0.001

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, drop_last=True)
test_loader = DataLoader(dataset=val_dataset, batch_size=batch_size)


#%%
class cancer_model(nn.Module):
    def __init__(self, in_channel = 3, hidden_channel = 32, out_channel = 2):
        super(cancer_model, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, hidden_channel, kernel_size = 3, stride = 1)
        self.conv2 = nn.Conv2d(hidden_channel, hidden_channel, kernel_size = 3, stride = 1)
        self.conv3 = nn.Conv2d(hidden_channel, hidden_channel, kernel_size = 3, stride = 1)
        self.batchnorm = nn.BatchNorm2d(hidden_channel)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.fc = nn.Linear(1176, out_channel)
    def forward(self, x):
        x = F.relu(self.batchnorm(self.conv1(x)))
        x = F.relu(self.batchnorm(self.conv2(x)))
        x = F.relu(self.batchnorm(self.conv3(x)))
        x = F.relu(self.pool(x))
        # print('size after pool', x.size())
        # print(type(x.size(0)))
        first_dim_size = x.size(0)
        x = x.reshape(first_dim_size, -1).contiguous()
        # first_dim_size = x.size(0)
        # print('size after research', x.size())
        x = nn.Linear(x.size(1), 2)(x)
        return F.sigmoid(x)
    
model = cancer_model()

#%%
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

train_epoch_loss = []
test_epoch_loss = []

for e in tqdm(range(1, epoch+1)):
    model.train()
    train_batch_loss = []
    test_batch_loss = []
    for x, y in train_loader:
        # print(len(x))
        x_batch = x.to(device)
        # print('x_batch:', x_batch.size())
        y_batch = y.to(device)
        # y_batch = one_hot_torch(y).to(device)
        # print('batch y size before flatten:',y_batch.size())
        # y_batch = y_batch.flatten()
        # print('batch y size after flatten:',y_batch.size())
        # print(x_batch.size())
        # print(x_batch.size())
# For example, if you have a convolutional layer with 64 output channels, 3 input channels, and a kernel size of 3x3, the weight parameters would have a dimension of (64, 3, 3, 3)
        # print(x_batch.size())
        # print('x_batch.size():',x_batch.size())
        pred = model(x_batch.float())
        loss_train = criterion(pred, y_batch)
        train_batch_loss.append(loss_train)
        loss_train.backward()
        optimizer.step()
        optimizer.zero_grad()
    train_epoch_loss.append(torch.mean(torch.stack(train_batch_loss)).detach().numpy())
    with torch.no_grad():
        # print('test')
        for x, y in test_loader:
            x_batch = x.to(device)
            y_batch = y.to(device)
            # print(x_batch.size())
            # y_batch = torch.Tensor.float(y).to(device)
            # x_batch = x_batch.permute(0, 3, 1, 2).to(device)
            pred = model(x_batch.float())
            loss_test = criterion(pred, y_batch)
            test_batch_loss.append(loss_test)
        test_epoch_loss.append(torch.mean(torch.stack(test_batch_loss)).detach().numpy())
    print(f'Epoch {e}')
    print(f"Training loss: {torch.mean(torch.stack(train_batch_loss)).detach().numpy()}")
    print(f"Validation loss: {torch.mean(torch.stack(test_batch_loss)).detach().numpy()}")
    print('==='*10)

#%%
fig, ax = plt.subplots()
x = np.arange(1, epoch+1, 1)
ax.plot(x, train_epoch_loss,label='Training')
ax.plot(x, test_epoch_loss,label='Validation')
ax.legend()
ax.set_xlabel("Number of Epoch")
ax.set_ylabel("Loss")
ax.set_xticks(np.arange(0, epoch+1, 10))
ax.set_title(f'Learning_rate:{lr}')
# ax_2 = ax.twinx()
# ax_2.plot(history["lr"], "k--", lw=1)
# ax_2.set_yscale("log")
# ax.set_ylim(ax.get_ylim()[0], history["training_losses"][0])
ax.grid(axis="x")
fig.tight_layout()
fig.show()
#%%
a = torch.zeros(1, 2, 3, 4, 5, 6)
b = a.view(a.shape[:2], -1, a.shape[5:])
