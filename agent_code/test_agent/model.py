import torch
import torch.nn as nn
import torch.nn.functional as F
import os

class LinearQNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # norm?
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        # depends on output
        x = self.linear3(x)
        return x

    def save(self):
        path = './my-saved-model.pt'
        torch.save(self.state_dict(), path)

    # def load(self, file_name='model.pth'):
    #     model_folder_path = './model'
    #     if not os.path.exists(model_folder_path):
    #         raise Exception("Could not load the model. No Model found")
    #     model.load_state_dict(torch.load(PATH))
    #     model.eval()
