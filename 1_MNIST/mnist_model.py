
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
    list_ = train_data_x.iloc[x,:].tolist()
    train_data_list.append(list_)
    
train_data_x = train_data_list
train_data_y = train_data.iloc[:,0].to_list()

# test_data_x = test_data
# test_data_list = []
# for x in range(test_data_x.shape[0]):
#     list_ = test_data_x.iloc[x,:].tolist()
#     test_data_list.append(list_)

# test_data_x = test_data_list

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


# train_dataset_x, val_dataset_x = random_split(train_data_x, [int(len(train_data_x)*0.8), len(train_data_x)-int(len(train_data_x)*0.8)])
# train_dataset_y, val_dataset_y = random_split(train_data_y, [int(len(train_data_y)*0.8), len(train_data_y)-int(len(train_data_y)*0.8)])

training_dataset = MNIST_Dataset(train_data_x, train_data_y) 

train_dataset, val_dataset = random_split(training_dataset, [int(len(training_dataset)*0.8), len(training_dataset)-int(len(training_dataset)*0.8)])

# testing_dataset = MNIST_Dataset(val_dataset_x, val_dataset_y ) 
#%%
def one_hot_torch(number):
    arr = torch.zeros((len(number), 10), dtype=torch.float32)
    for i, n in enumerate(number):
        arr[i, n] = 1
    return arr

# print(one_hot_torch([0,1,2,3]))
#%%
class Model(nn.Module):
    def __init__(self, input=784, hidden=20, out=10, dropout_rate=0.2):
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

#Training hyperparameters
epoch = 20
batch_size = 128
lr = 0.001

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size)
test_loader = DataLoader(dataset=val_dataset, batch_size=batch_size)
# criterion = nn.MSELoss()
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(train_linear_model.parameters(), lr=lr)

train_epoch_loss = []
test_epoch_loss = []

for e in tqdm(range(1, epoch+1)):
    train_linear_model.train()
    train_batch_loss = []
    test_batch_loss = []
    for x, y in train_loader:
        x_batch = torch.stack(x, dim=1).to(device)
        y_batch = one_hot_torch(y).to(device)
        # print(x_batch.size())
        pred = train_linear_model(x_batch.float())
        loss_train = criterion(pred, y_batch)
        train_batch_loss.append(loss_train)
        loss_train.backward()
        optimizer.step()
        optimizer.zero_grad()
    train_epoch_loss.append(torch.mean(torch.stack(train_batch_loss)).detach().numpy())
    with torch.no_grad():
        # print('test')
        for x, y in test_loader:
            x_batch = torch.stack(x, dim=1).to(device)
            y_batch = one_hot_torch(y).to(device)
            # print(x_batch.size())
            # y_batch = torch.Tensor.float(y).to(device)
            pred = train_linear_model(x_batch.float())
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

# %%
