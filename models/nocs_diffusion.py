import torch
from torch import nn


class FilmLayer(nn.Module):
    def __init__(self, in_dim,  out_dim, ctx_dim):
        self.proj = nn.Linear(in_dim, out_dim)
        self.scale = nn.Linear(ctx_dim, out_dim, bias=False)
        self.bias = nn.Linear(ctx_dim, out_dim)
    def forward(self, x):

class ContextualPointNet(nn.Module):
    def __init__(self, in_ch, )

class NocsDiff(nn.Module):
    def __init__(self):
        self.net = ...
        self.var_sched = ...

    def get_loss(self, data, context):
        