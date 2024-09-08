import torch
from torch import nn
from .linear_blocks import SetLinearHead, LinearHead 
from ..utils import ModelReturnType
from typing import List

class PointNetEncoder(nn.Module):
    def __init__(self, in_dim:int, layer_dims:List, out_dim:int):
        super().__init__()
        self.in_dim = in_dim
        self.latent_dims = layer_dims
        self.out_dim = out_dim

        self.point_proj = SetLinearHead([self.in_dim] + layer_dims, post_bn=True)
        self.mu_head    = LinearHead(layer_dims[::-1] + [self.out_dim])
        self.var_head   = LinearHead(layer_dims[::-1] + [self.out_dim])

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.point_proj(x)
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, self.latent_dims[-1])

        m = self.mu_head(x)[:, None]
        v = self.var_head(x)[:, None]

        # Returns both mean and logvariance, just ignore the latter in deteministic cases.
        return ModelReturnType(mu=m, var=v)

