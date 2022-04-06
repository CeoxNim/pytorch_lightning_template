import torch
from torch import nn

from einops import rearrange, reduce, repeat

class LinearModel(nn.Module):
    def __init__(self, num_hiddens):
        super(LinearModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(28 * 28, num_hiddens), 
            nn.ReLU(True),
            nn.Linear(num_hiddens, 10),
        )
    
    def forward(self, x):
        # Shape of 'x': ('bs', 1, 28, 28)
        x = rearrange(x, 'b 1 x y -> b (x y)', x=28, y=28)
        return self.net(x)