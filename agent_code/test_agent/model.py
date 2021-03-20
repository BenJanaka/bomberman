import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os


class LinearQNet(nn.Module):
    def __init__(self, output_size):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1),
            nn.ReLU()
        )

        self.linear1 = nn.Linear(64, 256, bias=True)
        self.linear2 = nn.Linear(256, output_size, bias=True)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = x.view((-1, 1, 17, 17))
        x = self.conv(x)
        x = x.view(-1, 64)
        x = torch.sigmoid(self.linear1(x))
        # x = self.dropout(x)
        x = self.linear2(x)
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
        state = torch.load(path)
        return state
