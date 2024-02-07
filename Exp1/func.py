import numpy as np
import torch
import torch.nn as nn
import  torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import sampler #
import torchvision.datasets as datasets
import torchvision.transforms as T
import matplotlib.pyplot as plt


if torch.cuda.is_available():
    print('GPU disponible')
    device = torch.device('cuda')
else:
    print('GPU no siponible, se hara uso del CPU')
    device = torch.device('cpu`')

def acccuracy(model, loader):
    num_correct = 0
    num_total = 0
    model.eval()
    model = model.to(device=device)
    with torch.no_grad():
        for x_i, y_i in loader:
            x_i = x_i.to(device=device, dtype = torch.float32)
            y_i = y_i.to(device=device, dtype= torch.long)
            scores = model(x_i)
            _, pred = scores.max(dim=1)
            num_correct += (pred == y_i).sum()
            num_total += pred.size(0)
        return float (num_correct)/num_total
    
def train(model, train_loader, val_loader, optimizer, epochs = 100):
    model = model.to(device=device)
    for epoch in range(epochs):
        for i, (x_i, y_i) in enumerate(train_loader):
            model.train()
            x_i = x_i.to(device=device, dtype = torch.float32)
            y_i = y_i.to(device=device, dtype= torch.long)
            scores = model(x_i)
            cost = F.cross_entropy(input=scores, target=y_i)
            optimizer.zero_grad()
            cost.backward()  #
            optimizer.step() #actualizacion de parametros
            acc = acccuracy(model=model, loader=val_loader)
            print(f'Epoch: {epoch}, costo: {cost.item()}, accuracy: {acc}')


