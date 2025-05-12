import sys
sys.path.append('/Users/amritnijjar/Desktop/mta_data_project_2/')

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torchinfo import summary

from datetime import datetime

from models.trainer import ModelTrainer

# Fully Connected Layers with station information

data_path = "/Users/amritnijjar/Desktop/mta_data_project_2/data/"

X_train = np.load(data_path + "X_train.npy")
y_train = np.load(data_path + "y_train.npy")

X_val = np.load(data_path + "X_val.npy")
y_val = np.load(data_path + "y_val.npy")

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

batch_size_ = 106

train_loader = DataLoader(train_dataset, batch_size = batch_size_, shuffle = True)
val_loader = DataLoader(val_dataset, batch_size = batch_size_, shuffle = False)

def construct_model(in_size, out_size, hidden_depth, hidden_width):

    layers = []

    layers.append(nn.Linear(in_size, hidden_width))
    layers.append(nn.ReLU())

    for i in range(hidden_depth):
        layers.append(nn.Linear(hidden_width, hidden_width))
        layers.append(nn.ReLU())

    layers.append(nn.Linear(hidden_width, out_size))

    return nn.Sequential(*layers)


for depth in [1, 2, 3]:
    for width in [24, 48, 96]:

        print(f"Depth: {depth}, Width: {width}, Batch: {batch_size_}")

        model_ = construct_model(in_size = 26, out_size = 6, hidden_depth = depth, hidden_width = width)
        optimizer_ = optim.Adam(model_.parameters(), lr = 0.001)
        loss_fn_ = nn.L1Loss()
        n_epochs_ = 50

        trainer = ModelTrainer(model = model_, n_epochs = n_epochs_, optimizer = optimizer_, loss_fn = loss_fn_, save_loc = "/Users/amritnijjar/Desktop/mta_data_project_2/models/FF_with_station/")

        trainer.train(train_loader, val_loader)

        trainer.save(f"FF_with_station_depth{depth}_width{width}_batch{batch_size_}")

    
    






