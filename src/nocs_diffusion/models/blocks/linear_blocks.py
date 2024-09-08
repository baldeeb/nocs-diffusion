import torch
from torch import nn

from ..utils import ModelReturnType

class LinearHead(nn.Module):
    def __init__(self, dims, activation=nn.ReLU(), post_bn=False):
        super().__init__()
        assert len(dims) > 1, "LinearHead handles only 2 or more layers."
        layers = []        
        for d_in, d_out in zip(dims[:-2], dims[1:-1]):
            layers.append(nn.Linear(d_in, d_out))
            layers.append(nn.BatchNorm1d(d_out))
            layers.append(activation)
        layers.append(nn.Linear(dims[-2], dims[-1]))
        if post_bn:
            layers.append(nn.BatchNorm1d(dims[-1]))
            
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class SetLinearHead(nn.Module):
    def __init__(self, dims, activation=nn.ReLU(), post_bn=False):
        super().__init__()
        assert len(dims) > 1, "SetLinearHead handles only 2 or more layers."
        layers = []
        for d_in, d_out in zip(dims[:-2], dims[1:-1]):
            layers.append(nn.Conv1d(d_in, d_out, 1))
            layers.append(nn.BatchNorm1d(d_out))
            layers.append(activation)
        layers.append(nn.Conv1d(dims[-2], dims[-1], 1))
        if post_bn:
            layers.append(nn.BatchNorm1d(dims[-1]))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)