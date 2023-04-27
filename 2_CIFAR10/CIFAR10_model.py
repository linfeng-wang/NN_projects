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
from torch.autograd import variable
from itertools import chain
from sklearn import metrics as met
import pickle
from icecream import ic

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

# %%
# train_loader = torch.utils.data.DataLoader(
#     datasets.CIFAR10('./data', train = True, download=True,
#     transform = transforms.Compose([transforms.ToTensor(),
#                                     transforms.Normalize((0.1307),(0.3081,))])),
#     batch_size = 64, shuffle = True)

# test_loader = torch.utils.data.DataLoader(
#     datasets.CIFAR10('./data', train = False, download=True,
#     transform = transforms.Compose([transforms.ToTensor(),
#                                     transforms.Normalize((0.1307),(0.3081,))])),
#     batch_size = 64, shuffle = True)

# %%
device = 'cuda' if torch.cuda.is_available() else 'cpu'

my_dict = {'frog': 0, 'truck': 1, 'deer': 2, 'automobile': 3, 'bird': 4, 'horse': 5, 'ship': 6, 'cat': 7, 'dog': 8, 'airplane': 9}
def one_hot_torch(y):
    arr = torch.zeros((10), dtype=torch.float32)
    arr[my_dict[y]] = 1.
    return arr

trainLabels = pd.read_csv('data/cifar-10/trainLabels.csv')

train_x = trainLabels['id']
train_y = trainLabels['label']

class cifar10_dataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __getitem__(self, index):
        i = index + 1
        img = Image.open(f'data/cifar-10/train/{i}.png')
        arr = np.array(img) # 32x32x3 array
        return arr, one_hot_torch(self.y[index])
    def __len__(self):
        return len(self.x)
    

training_dataset = cifar10_dataset(train_x, train_y) 

train_dataset, val_dataset = random_split(training_dataset, [int(len(training_dataset)*0.8), len(training_dataset)-int(len(training_dataset)*0.8)])

#%%
class Model(nn.Module):
    def __init__(self, in_channel = 3, hidden_channel = 6, out_channel=10):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, hidden_channel, kernel_size=3)
        self.conv2 = nn.Conv2d(hidden_channel, hidden_channel, kernel_size=3)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(1176, out_channel)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.pool(x))
        # print('size after pool', x.size())
        # print(type(x.size(0)))
        first_dim_size = x.size(0)
        x = x.reshape(first_dim_size, -1).contiguous()
        # first_dim_size = x.size(0)
        # print('size after research', x.size())
        x = self.fc(x)
        return F.sigmoid(x)

model = Model()


epoch = 20
batch_size = 128
lr = 0.001

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size)
test_loader = DataLoader(dataset=val_dataset, batch_size=batch_size)
# criterion = nn.MSELoss()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

train_epoch_loss = []
test_epoch_loss = []

for e in tqdm(range(1, epoch+1)):
    model.train()
    train_batch_loss = []
    test_batch_loss = []
    for x, y in train_loader:
        x_batch = x.to(device)
        y_batch = y.to(device)
        # y_batch = one_hot_torch(y).to(device)
        # print('batch y size before flatten:',y_batch.size())
        # y_batch = y_batch.flatten()
        # print('batch y size after flatten:',y_batch.size())
        # print(x_batch.size())
        x_batch = x_batch.permute(0, 3, 1, 2).to(device)
        # print(x_batch.size())
# For example, if you have a convolutional layer with 64 output channels, 3 input channels, and a kernel size of 3x3, the weight parameters would have a dimension of (64, 3, 3, 3)
        # print(x_batch.size())
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
            x_batch = x_batch.permute(0, 3, 1, 2).to(device)
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
# %%