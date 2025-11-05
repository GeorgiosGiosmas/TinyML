import torch
from torch import nn
from torch.optim import Adam
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import pandas as pd
import os

def evaluate(model: nn.module, dataloader: Dataloader):
    
    total_loss_test = 0
    total_acc_test = 0
    inference_time = []
    
     with torch.no_grad():
        for images, labels in tqdm(dataloader):
            outputs = model(images)
            val_loss = criterion(outputs, labels)
            total_loss_val += val_loss.item()
            
            val_acc = (torch.argmax(outputs, axis=1) == labels).sum().item()
            total_acc_val += val_acc
            
        epoch_finish = round((time.time() - epoch_start)/60, 2)


if __name__ == '__main__':
 
    pass