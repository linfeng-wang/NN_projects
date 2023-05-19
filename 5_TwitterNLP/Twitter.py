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
import itertools
import matplotlib.pyplot as plt
import pathlib
from sklearn.model_selection import train_test_split
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn import feature_extraction, linear_model, model_selection, preprocessing
import string
from importlib import reload
# import util
# import model_torch_simple
# from torchmetrics import Accuracy
from tqdm import tqdm
import argparse

import numpy as np
from PIL import Image

#%%
np.random.seed(3)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
df = pd.read_csv('nlp-getting-started/train.csv')
# randomly shuffle the rows in the DataFrame
df = df.sample(frac=1).reset_index(drop=True)

# count_vectorizer = feature_extraction.text.CountVectorizer()
# count_vectorizer.fit(df['text'])

#%%
embeddings_index = {}
with open('glove.6B.50d.txt','r',encoding = 'utf8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.array(values[1:],'float32')
        embeddings_index[word] = coefs
        
print('Found %s word vectors.' % len(embeddings_index))


#%%

def clean_text(text):
    #2. remove unkonwn characrters
    emoji_pattern = re.compile("["
                    u"\U0001F600-\U0001F64F"  # emoticons
                    u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                    u"\U0001F680-\U0001F6FF"  # transport & map symbols
                    u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                    u"\U00002702-\U000027B0"
                    u"\U000024C2-\U0001F251"
                    "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)

    #1. remove http links
    url = re.compile(r'https?://\S+|www\.\S+')
    text = url.sub(r'',text) # 
    
    #3,4. remove #,@ and othet symbols
    text = text.replace('#',' ')
    text = text.replace('@',' ')
    text = text.replace('  ',' ')

    symbols = re.compile(r'[^A-Za-z0-9 ]')
    text = symbols.sub(r'',text)
    
    #5. lowercase
    text = text.lower()
    
    return text

df['text'] = df['text'].apply(lambda x: clean_text(x))

#%%
new_embedding = {}

embed_keys = embeddings_index.keys()
for x in tqdm(df['text']):
    list1 = x.split(' ')
    new_list = []
    for i in list1:
        if((i in embed_keys)  and (i not in new_embedding.keys())):
            new_embedding[i] = embeddings_index[i]
            
        elif((i not in embeddings_index) and (i not in new_embedding.keys())):
            new_embedding[i] = np.random.normal(scale=0.4, size=(50, ))

        else:
            continue
new_embedding['<pad>'] = np.zeros(50)

#%%
max_len = 0 #=50
for x in df['text']:
    if len(x.split(' ')) > max_len:
        max_len = len(x.split(' '))

#%%
def pad_features(reviews_int, max_len=max_len):
    review_len = len(reviews_int.split(' '))
    padding = ' <pad>' * (max_len-review_len)
    review_out = reviews_int + padding
    return review_out


#%%
class nlp_dataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
        self.X = self.df['text']
        self.y = self.df['target']
        self.len = len(self.df)
        # self.onehot = OneHotEncoder()
        # self.onehot.fit(self.y.to_numpy().reshape(-1,1))
        # self.y_onehot = self.onehot.transform(self.y.to_numpy().reshape(-1,1)).toarray()
        # self.y_onehot = torch.tensor(self.y_onehot).float()
    def __getitem__(self, index):
        padded_x = pad_features(self.X[index])
        text = []
        for word in padded_x.split(' '): text.append(torch.tensor(new_embedding[word]))
        target = torch.tensor(int(self.y[index]))
        return torch.stack(text), target
    def __len__(self):
        return self.len

train_dataset = nlp_dataset(df)
train_data, val_data = random_split(train_dataset, [int(len(train_dataset)*0.8), len(train_dataset)-int(len(train_dataset)*0.8)])

#%%
class nlp_model(nn.Module):
    def __init__(self, input_size = 50, hidden_size = 128, output_size = 2):
        super(nlp_model, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
    def forward(self, input_tensor, hidden_tensor):
        # print(input_tensor[i].size(), hidden_tensor.size())
        # input_tensor = torch.tensor(input_tensor,requires_grad=True)
        # hidden_tensor = torch.tensor(hidden_tensor,requires_grad=True)
        combinations = torch.cat((input_tensor, hidden_tensor), 1)
        combinations = combinations.to(torch.float32)
        # combinations = torch.tensor(combinations,requires_grad=True)
        # print(combinations.size())
        hidden = self.i2h(combinations)
        output = self.i2o(combinations)
        output = self.softmax(output)
        # print('output_size:',output.size())
        return output, hidden
    
    def init_hidden(self):
        return torch.zeros(1, self.hidden_size,)

rnn = nlp_model()


 #%%
# input_size = 50 
# hidden_size = 50
# output_size = 2
# for input_tensor,y in train_loader:
#     input_tensor = input_tensor.permute(1,0,2)
#     # print(input_tensor.size())
    
#     hidden_size = hidden_size
#     i2h = nn.Linear(input_size + hidden_size, hidden_size)
#     i2o = nn.Linear(input_size + hidden_size, output_size)
#     softmax = nn.LogSoftmax(dim=1)
    
#     for i in range(len(input_tensor)):
#         # print(input_tensor[i].size(), hidden_tensor.size())
#         combinations = torch.cat((input_tensor[i], hidden_tensor), 1)
#         # print(combinations.size())

#         # combinations = combinations.transpose(0, 1)
#         # print(combinations.size())

#         combinations = combinations.to(torch.float32)
#         # print(combinations)
#         # print(combinations.size())
#         # print(combinations.size())
#         hidden = i2h(combinations)
#         output = i2o(combinations)
#         output = softmax(output)
#     print(output.size())
#     break 

#%%
# class nlp_model(nn.Module):
#     def __init__(self, input_size = 50, hidden_size = 50, output_size = 2):
#         super(nlp_model, self).__init__()
#         self.hidden_size = hidden_size
#         self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
#         self.i2o = nn.Linear(input_size + hidden_size, output_size)
#         self.softmax = nn.LogSoftmax(dim=1)
#     def forward(self, input_tensor, hidden_tensor):
#         for i in range(len(input_tensor)):
#             # print(input_tensor[i].size(), hidden_tensor.size())
#             combinations = torch.cat((input_tensor[i], hidden_tensor), 1)
#             # print(combinations.size())
#             hidden = self.i2h(combinations)
#             output = self.i2o(combinations)
#             output = self.softmax(output)
#         print('output_size:',output.size())
#         return output
    
#     def init_hidden(self):
#         return torch.zeros(self.hidden_size,50)

# rnn = nlp_model()
#%%
lr = 0.005
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(rnn.parameters(), lr=lr)
batch_size = 1
n_epoch = 20
train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset=val_data, batch_size=batch_size, shuffle=True)

#%%
# c = 0
# while c <= 2:
for x, y in train_loader:
    print(x.size())
    print(y.size())
    break
#%%
train_epoch_loss = []
test_epoch_loss = []

for e in tqdm(range(1, n_epoch+1)):
    rnn.train()
    train_batch_loss = []
    test_batch_loss = []
    for x, y in train_loader:
        
        # print(len(x))
        x_batch = x.to(device)
        x_batch = x.squeeze(0)
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
        hidden = rnn.init_hidden()
        for i in range(x_batch.size()[0]):
            output, hidden = rnn(x_batch[i].unsqueeze(0), hidden)

        loss_train = criterion(output, y_batch)
        train_batch_loss.append(loss_train)
        # loss_train = torch.tensor(loss_train, requires_grad=True)
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()
    train_epoch_loss.append(torch.mean(torch.stack(train_batch_loss)).detach().numpy())
    with torch.no_grad():
        # print('test')
        for x, y in val_loader:
            x_batch = x.to(device)
            x_batch = x.squeeze(0)
            y_batch = torch.tensor(y).to(device)
            # print(x_batch.size())
            hidden = rnn.init_hidden()
            for i in range(x_batch.size()[0]):
                output, hidden = rnn(x_batch[i].unsqueeze(0), hidden)

            # y_batch = torch.Tensor.float(y).to(device)
            # x_batch = wha.to(device)
            loss_test = criterion(output, y_batch)
            
            test_batch_loss.append(loss_test)
        test_epoch_loss.append(torch.mean(torch.stack(test_batch_loss)).detach().numpy())
    print(f'Epoch {e}')
    print(f"Training loss: {torch.mean(torch.stack(train_batch_loss)).detach().numpy()}")
    print(f"Validation loss: {torch.mean(torch.stack(test_batch_loss)).detach().numpy()}")
    print('==='*10)

#%%
fig, ax = plt.subplots()
x = np.arange(1, n_epoch+1, 1)
ax.plot(x, train_epoch_loss,label='Training')
ax.plot(x, test_epoch_loss,label='Validation')
ax.legend()
ax.set_xlabel("Number of Epoch")
ax.set_ylabel("Loss")
ax.set_xticks(np.arange(0, n_epoch+1, 10))
ax.set_title(f'Learning_rate:{lr}')
# ax_2 = ax.twinx()
# ax_2.plot(history["lr"], "k--", lw=1)
# ax_2.set_yscale("log")
# ax.set_ylim(ax.get_ylim()[0], history["training_losses"][0])
ax.grid(axis="x")
fig.tight_layout()
fig.show()
#%%
for x, y in train_loader:
    print(x.size())




# %%
