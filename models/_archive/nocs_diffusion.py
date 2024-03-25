import torch
from torch import nn
from torch import Tensor
from torch.nn import Module
import torch.nn.functional as F

from torch.nn import Conv2d
from torchvision.models import resnet18
from torchvision.transforms import Resize
from torchvision.models.segmentation import deeplabv3_resnet50

from models.context_encoder import CtxtEncoder
from models.scheduler import VarianceSchedule
from .blocks import FilmResLayer

from typing import Optional

class ForwardDiffuser(nn.Module):
    def __init__(self, var_sched:VarianceSchedule, mean:float=0.0):
        super().__init__()
        self.var_sched = var_sched
        self._mu = mean
    def __call__(self, x:Tensor, noise:Optional[Tensor]=None, t:Optional[int]=None):  
        if noise is None:
            noise = torch.randn_like(x, device=x.device)
        s, n = self.var_sched.sample(noise.shape[0], t)
        s = s[:, None, None, None].to(x.device) 
        n = n[:, None, None, None].to(x.device)
        return x*s + (noise + self._mu)*n
    
class BackwardDiffuser(nn.Module):
    def __init__(self, diffuser:nn.Module):
        super().__init__()
        self._diff = diffuser


class NocsDiff(Module):
    def __init__(self, in_dim, ctx_encoder, in_plane):
        super().__init__()
        self.ctx_net = ctx_encoder
        ctx_dim = self.ctx_net.out_dim
        
        # self.img_proj = nn.Sequential(
        #     nn.Conv2d(in_dim, in_plane, 7, 1, 3),
        #     nn.BatchNorm2d(in_plane),
        #     nn.GELU(),
        # )
        self.img_proj = nn.Sequential(
            nn.Conv2d(in_dim, in_plane, 1, 1, 0),
            nn.BatchNorm2d(in_plane),
            nn.GELU(),
        )
        self.img_net = nn.ModuleDict({
            'l1' : FilmResLayer(in_plane, in_plane, ctx_dim),
            'l2' : FilmResLayer(in_plane, in_plane, ctx_dim),
            'l3' : FilmResLayer(in_plane, in_plane, ctx_dim),
            'l4' : FilmResLayer(in_plane, in_dim,   ctx_dim),
        })
        self.temp_ctxt = None

    def forward(self, x, ctx):
        
        # TESTING: Temporarily I am using the same image so depth does not vary.
        if self.temp_ctxt is None:
            self.temp_ctxt = torch.randn((x.shape[0], self.ctx_net.out_dim), device=x.device)
        ctx = self.temp_ctxt

        # ctx = self.ctx_net(ctx)
        x = self.img_proj(x)
        for layer in self.img_net.values():
            x = layer(x, ctx)
        return x

    def get_loss(self, images:Tensor, context:Tensor):
        noise = torch.randn_like(images, device=images.device)
        pred_noise = self.forward(images, context)
        loss = F.mse_loss(noise, pred_noise)
        return loss


if __name__ == '__main__':
    
    depth = torch.rand(2, 1, 64,64)
    img = torch.rand(2, 3, 64,64)
    
    ctx_net = CtxtEncoder(1, 64)
    diffuser = NocsDiff(3, ctx_net)

    dx = diffuser(img, depth)

    pass