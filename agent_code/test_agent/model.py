import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys


class LinearQNet(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()

        self.path = 'my-saved-model.pt'

        self.linear1 = nn.Linear(input_size, 6, bias=False)
        self.linear2 = nn.Linear(6, 6, bias=False)
        self.linear3 = nn.Linear(6, 6, bias=False)
        self.linear4 = nn.Linear(6, 6, bias=False)
        self.linear5 = nn.Linear(6, output_size, bias=False)
        self.dropout = nn.Dropout(0.)
        # dropout example:
        # https://wandb.ai/authors/ayusht/reports/Dropout-in-PyTorch-An-Example--VmlldzoxNTgwOTE

    def forward(self, x):
        # norm?
        x = torch.sigmoid(self.linear1(x))
        # x = self.dropout(x)
        x = torch.sigmoid(self.linear2(x))
        # x = self.dropout(x)
        # x = torch.relu(self.linear3(x))
        # x = self.dropout(x)
        # x = torch.sigmoid(self.linear4(x))
        # x = self.dropout(x)
        x = self.linear5(x)

        return x # output must be in same order of mag as rewards -> no activation for ouput

    def save(self, optimizer, score):
        state = {
            'score': score,
            'model': self.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        torch.save(state, self.path)

    def load(self) -> dict:
        if not os.path.exists(self.path):
            raise Exception("Could not load the model. No Model found")
        state = torch.load(self.path)
        return state
