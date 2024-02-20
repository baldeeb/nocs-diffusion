import torch
from torch import nn
from torch.nn import Module, Identity, Conv2d
from torchvision.models import resnet18
from torchvision.transforms import Resize
from torchvision.models.segmentation import deeplabv3_resnet50

class CtxtEncoder(Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.in_dim, self.out_dim = in_dim, out_dim
        self._net = resnet18()
        self._net.conv1 = Conv2d(in_dim, 64, (7, 7), (2, 2), (3, 3), bias=False)
        self._net.fc = nn.Linear(512, out_dim)

    def forward(self, x):
        return self._net(x)


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


class NocsDiff(Module):
    def __init__(self, in_dim, ctx_encoder, in_plane=32):
        super().__init__()
        self.ctx_net = ctx_encoder
        ctx_dim = self.ctx_net.out_dim
        
        self.img_proj = nn.Sequential(
            nn.Conv2d(in_dim, in_plane, 7, 1, 3),
            nn.BatchNorm2d(in_plane),
            nn.GELU(),
        )
        self.img_net = nn.ModuleDict({
            'l1' : FilmResLayer(in_plane, in_plane, ctx_dim),
            'l2' : FilmResLayer(in_plane, in_plane, ctx_dim),
            'l3' : FilmResLayer(in_plane, in_plane, ctx_dim),
            'l4' : FilmResLayer(in_plane, in_dim,   ctx_dim),
        })

    def forward(self, x, ctx):
        ctx = self.ctx_net(ctx)
        x = self.img_proj(x)
        for layer in self.img_net.values():
            x = layer(x, ctx)
        return x

if __name__ == '__main__':
    
    depth = torch.rand(2, 1, 64,64)
    img = torch.rand(2, 3, 64,64)
    
    ctx_net = CtxtEncoder(1, 64)
    diffuser = NocsDiff(3, ctx_net)

    dx = diffuser(img, depth)

    pass