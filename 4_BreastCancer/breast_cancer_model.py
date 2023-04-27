
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

# %%
all_samples = glob.glob('data/archive/*')

def imshow(inp, title): # inp is a tensor
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0)) # convert to numpy array
    inp = std * inp + mean # unnormalize
    inp = np.clip(inp, 0, 1) # clip to 0-1
    plt.imshow(inp)
    plt.title(title)
    plt.show()

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
def extract_class_number(input_string):
    pattern = r'class(\d+).png'  # regex pattern to match the class number
    match = re.search(pattern, input_string)
    if match:
        # print(match.group(0))
        # print(match.group(1))
        class_number = int(match.group(1))
        return class_number
    else:
        return None

class_number = [extract_class_number(x) for x in file_paths]
# print(extract_class_number('/Users/linfengwang/Github/NN_projects/4_BreastCancer/data/archive/8863/0/8863_idx5_x51_y1251_class192038920.png')) # testing

# %%
class breast_cancer_dataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __getitem__(self, index):
        img = read_image(self.x[index])
        class_type = self.y[index]
        return img, class_type
    def __len__(self):
        return len(self.x)

training_dataset = breast_cancer_dataset(file_paths, class_number)

train_dataset, val_dataset = random_split(training_dataset, [int(len(training_dataset)*0.8), len(training_dataset)-int(len(training_dataset)*0.8)])


#
# %%
