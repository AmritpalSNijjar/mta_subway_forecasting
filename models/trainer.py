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

class ModelTrainer:
    
    def __init__(self, model, n_epochs, optimizer, loss_fn, save_loc):
        
        self.trained = False
        
        self.model = model
        self.n_epochs = n_epochs
        
        self.optimizer = optimizer
        self.loss_fn = loss_fn

        self.save_loc = save_loc

    def train(self, train_loader, val_loader):
        
        summary(self.model)
        print("Beginning training...")

        for epoch in range(self.n_epochs):
            start_time = datetime.now()

            # Training
            self.model.train()
            train_loss = 0
            
            for batch_idx, (batch_X, batch_y) in enumerate(train_loader):
                
                y_pred = self.model(batch_X)
                loss = self.loss_fn(batch_y, y_pred)
    
                train_loss += loss.item()
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
    
            train_loss /= len(train_loader)
    
            # Validating
            with torch.no_grad():
                
                self.model.eval()
                val_loss = 0
                
                for batch_idx, (batch_X, batch_y) in enumerate(val_loader):
        
                    y_pred = self.model(batch_X)
                    val_loss += self.loss_fn(batch_y, y_pred).item()
                
                val_loss /= len(val_loader)
            
            end_time = datetime.now()
            if epoch == 0 or (epoch + 1) % 10 == 0:
                print(f"Epoch: {epoch+1:>3}: Train Loss: {train_loss:.4f}: Val Loss: {val_loss:.4f}: Time: {(end_time - start_time).total_seconds():.2f}")
        
        print("Training complete!")
        self.trained = True

    def predict(self, X):
        pass
        
    def save(self, filename):
        torch.save(self.model.state_dict(), self.save_loc + filename + ".pth")
        print("Model saved as " + filename + ".")
        
    def load(self, filename):
        self.trained = True
        torch.load_state_dict(torch.load(self.save_loc + filename, weights_only=True))
        print("Model Loaded!")



    