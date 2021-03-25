import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

class LinearQNet(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=7, out_channels=16, kernel_size=7, stride=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=2, stride=2),
            nn.ReLU(),
        )
        self.linear1 = nn.Linear(32, 256)
        self.linear2 = nn.Linear(256, output_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, tau=None):
        x = x.view((-1, 7, 29, 29))
        x = self.conv(x)
        x = x.view((-1, 32))
        x = torch.sigmoid(self.linear1(x))
        # x = self.dropout(x)
        x = self.linear2(x)

        # return probabilities for exploration
        if tau:
            x = self.softmax_exploration(x, tau)
        return x # output must be in same order of mag as rewards -> no activation for output

    def save(self, optimizer, score, path):
        state = {
            'score': score,
            'model': self.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        torch.save(state, path)

    def load(self, path) -> dict:
        if not os.path.exists(path):
            raise Exception("Could not load the model. No Model found")
        if not torch.cuda.is_available():
            state = torch.load(path, map_location='cpu')
        else:
            state = torch.load(path)

        return state

    def softmax_exploration(self, Q, tau):
        prob = torch.exp(Q / tau)
        return prob / torch.sum(prob)
