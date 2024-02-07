
import torch
import torch.nn as nn
import torch.nn.functional as F

class Red(nn.Module):
    
    def __init__(self, n_entradas):
        super(Red, self).__init__()
        self.linear1 = nn.Linear(n_entradas, 15)
        self.linear2 = nn.Linear(15, 8)
        self.linear3 = nn.Linear(8,160)
        self.linear4 = nn.Linear(160, 200)
        self.linear5 = nn.Linear(200, 8)
        self.linear6 = nn.Linear(8, 1)
    
    def forward(self, inputs):
        pred_1 = torch.sigmoid(input=self.linear1(inputs))
        pred_2 = torch.sigmoid(input=self.linear2(pred_1))
        pred_3 = torch.sigmoid(input=self.linear3(pred_2))
        pred_4 = torch.sigmoid(input=self.linear4(pred_3))
        pred_5 = torch.sigmoid(input=self.linear5(pred_4))
        pred_f = torch.sigmoid(input=self.linear6(pred_5))
        return pred_f

