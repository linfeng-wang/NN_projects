
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
# import icecream as ic

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
# %%
# train_loader = torch.utils.data.DataLoader(
#     datasets.MNIST('./data', train = True, download=True,
#     transform = transforms.Compose([transforms.ToTensor(),
#                                     transforms.Normalize((0.1307),(0.3081,))])),
#     batch_size = 64, shuffle = True)

# test_loader = torch.utils.data.DataLoader(
#     datasets.MNIST('./data', train = False, download=True,
#     transform = transforms.Compose([transforms.ToTensor(),
#                                     transforms.Normalize((0.1307),(0.3081,))])),
#     batch_size = 64, shuffle = True)

# %%
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = 'cuda' if torch.cuda.is_available() else 'cpu'

train_data = pd.read_csv('/Users/linfengwang/Github/NN_projects/1_MNIST/data/MNIST/digit-recognizer/train.csv')
test_data = pd.read_csv('/Users/linfengwang/Github/NN_projects/1_MNIST/data/MNIST/digit-recognizer/test.csv')

train_data_x = train_data.iloc[:,1:]
train_data_list = []
for x in range(train_data_x.shape[0]):
    list_ = train_data_x.iloc[x,1:].to_list()
    train_data_list.append(list_)
    
train_data_x = train_data_list
train_data_y = train_data.iloc[:,1].to_list()

test_data_x = test_data.iloc[:,1:]
test_data_list = []
for x in range(test_data_x.shape[0]):
    list_ = test_data_x.iloc[x,1:].to_list()
    test_data_list.append(list_)

test_data_x = test_data_list
test_data_y = test_data.iloc[:,1].to_list()

# %%1_MNIST/data/MNIST/digit-recognizer/test.csv
class MNIST_Dataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.len = len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]
    def __len__(self):
        return self.len

training_dataset = MNIST_Dataset(train_data_x, train_data_y) 
testing_dataset = MNIST_Dataset(test_data_x, test_data_y) 

#%%
class Model(nn.Module):
    def __init__(self, input=1, hidden=20, out=10, dropout_rate=0.2):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(input, hidden, kernel_size = 5)
        self.conv2 = nn.Conv2d(hidden, hidden, kernel_size = 5)
        self.fc = nn.Linear(hidden, out)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size = 2)
        x = x.squeeze(dim = -1)
        x = self.fc(x)
        return F.sigmoid(x)
    
train_cnn_model = Model()

class Model(nn.Module):
    def __init__(self, input=1, hidden=20, out=10, dropout_rate=0.2):
        super(Model, self).__init__()
        self.lin1 = nn.Linear(input, hidden)
        self.lin2 = nn.Linear(hidden, hidden)
        self.fc = nn.Linear(hidden, out)
        
    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        # x = F.max_pool2d(x, kernel_size = 2)
        # x = x.squeeze(dim = -1)
        x = self.fc(x)
        return F.sigmoid(x)
    
train_linear_model = Model()

#%%
epoch = 10
batch_size = 128
lr = 0.001

train_loader = DataLoader(dataset=training_dataset, batch_size=batch_size)
test_loader = DataLoader(dataset=testing_dataset, batch_size=batch_size)
loss = nn.MSELoss()
optimizer = torch.optim.Adam(train_linear_model.parameters(), lr=lr)

train_epoch_loss = []
test_epoch_loss = []

for x in tqdm(range(1, epoch+1)):
    train_linear_model.train()
    train_batch_loss = []
    test_batch_loss = []
    for x, y in train_loader:
        pred = train_linear_model(torch.FloatTensor(x))
        loss_train = loss(y - pred)
        train_batch_loss.append(loss_train)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    train_epoch_loss.append(np.mean(train_batch_loss))
    with torch.no_grad():
        for x, y in test_loader:
            pred = train_linear_model(x)
            loss_test = loss(y-pred)
            test_epoch_loss.append(test_batch_loss)
        
#%%
for x, y in train_loader:
    x_batch = torch.stack(x)
    print(x_batch.size())
    break

# %%
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
