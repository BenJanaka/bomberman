import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os


class LinearQNet(nn.Module):
    def __init__(self, output_size):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=7, out_channels=64, kernel_size=9, padding=2), #größeren Kernel, padding
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=7),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5),
            nn.MaxPool2d(2),
            # nn.GroupNorm(4, 128),
            nn.ReLU(),
        )
        self.linear1 = nn.Linear(2*2*256, 256)
        self.linear2 = nn.Linear(256, output_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = x.view((-1, 7, 29, 29))
        x = self.conv(x)
        # print(x.shape)
        x = x.view((-1, 2 * 2 * 256))
        x = torch.sigmoid(self.linear1(x))
        x = self.dropout(x)
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
        if not torch.cuda.is_available():
            state = torch.load(path, map_location='cpu')
        else:
            state = torch.load(path)

        return state
