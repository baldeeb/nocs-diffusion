import torch
from torch import nn
from torch import Tensor
from torch.nn import Module, Conv2d
from torchvision.models import resnet18

class FilmResLayer(nn.Module):
    def __init__(self, in_dim, out_dim, ctx_dim):
        super().__init__()

        self.conv1 = Conv2d(in_dim, out_dim, 3, 1, 1)
        self.activ1 = nn.GELU()

        
        self.conv2 = Conv2d(out_dim, out_dim, 1)
        self.bn2 = nn.BatchNorm2d(out_dim)
        
        self.scale = nn.Linear(ctx_dim, out_dim, bias=False)
        self.bias = nn.Linear(ctx_dim, out_dim)

        if in_dim != out_dim:
            self.res_proj = nn.Conv2d(in_dim, out_dim, 1)
        else:
            self.res_proj = nn.Identity()
        
        self.activ2 = nn.GELU()
        
    def forward(self, x, ctx):
        r = self.conv1(x)
        r = self.activ1(r)

        r = self.conv2(r)
        r = self.bn2(r)

        s = self.scale(ctx)[:, :, None, None]
        b = self.bias(ctx)[:, :, None, None]
        r = r*s  + b
        r = self.activ2(r)

        r = r + self.res_proj(x)
        return r
