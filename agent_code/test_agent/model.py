import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os


class LinearQNet(nn.Module):
    def __init__(self, output_size):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=32, kernel_size=(5, 5), stride=1, padding=1, bias=True),
            nn.GroupNorm(num_groups=4, num_channels=32),
            nn.CELU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5, 5), stride=2, bias=True),
            nn.GroupNorm(num_groups=4, num_channels=64),
            nn.CELU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=1, bias=True),
            nn.GroupNorm(num_groups=4, num_channels=128),
            nn.CELU()
        )
        self.linear1 = nn.Linear(4*4*128, 512, bias=True)
        self.linear2 = nn.Linear(512, output_size, bias=True)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = x.view(-1, 2, 17,17)
        x = self.conv(x)
        x = x.view(-1, 4*4*128)
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
