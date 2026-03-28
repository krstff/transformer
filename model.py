import torch.nn as nn
import torch
import math
from einops import rearrange
import config

class FeedForward(nn.Module):
    def __init__(self, input_dim, num_neurons):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_neurons)
        self.activaiton = nn.GELU()
    
    def forward(self, x):
        # xi = x * W + b
        y = self.linear(x)
        # y = σ(xi)
        y = self.activaiton(y)

        return y
