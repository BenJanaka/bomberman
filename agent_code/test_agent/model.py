import torch
import torch.nn as nn
import torch.nn.functional as F
import os


class LinearQNet(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, 300)
        self.linear2 = nn.Linear(300, 150)
        self.linear3 = nn.Linear(150, 80)
        self.linear4 = nn.Linear(80, 40)
        self.linear5 = nn.Linear(40, output_size)
        self.dropout = nn.Dropout(0.25)
        # dropout example:
        # https://wandb.ai/authors/ayusht/reports/Dropout-in-PyTorch-An-Example--VmlldzoxNTgwOTE

    def forward(self, x):
        # norm?
        x = torch.sigmoid(self.linear1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.linear2(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.linear3(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.linear4(x))
        x = self.dropout(x)
        x = F.softmax(self.linear5(x), dim=0) # F.Softmax erlaubt nur outputs zwischen 0 und 1
        return x

    def save(self):
        path = 'my-saved-model.pt'
        torch.save(self.state_dict(), path)

    # def load(self, file_name='model.pth'):
    #     model_folder_path = './model'
    #     if not os.path.exists(model_folder_path):
    #         raise Exception("Could not load the model. No Model found")
    #     model.load_state_dict(torch.load(PATH))
    #     model.eval()
