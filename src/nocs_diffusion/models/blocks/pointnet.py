import torch
from torch import nn
from .linear_blocks import SetLinearHead, LinearHead 
from ..utils import ModelReturnType
from typing import List, Optional

class PointProjector(nn.Module):
    '''A PointNet minus the TNet'''
    def __init__(self, dims:List[int], variational:bool=False, \
                 last_proj_dims:List[int]=None, pooling:str="max"):
        super().__init__()
        self.dims = dims 
        self.last_proj_dims = last_proj_dims
        self.pooling = pooling
        
        if len(self.dims) <= 2: 
            self.point_proj = nn.Identity()
        else:
            self.point_proj = SetLinearHead(self.dims[:-1], post_bn=True)
        
        self.mu_head = self.var_head = None
        if last_proj_dims:
            self.mu_head    = LinearHead(last_proj_dims)
            if variational:
                self.var_head   = LinearHead(last_proj_dims)

    def forward(self, x):
        ''' Expects a cloud of (Batch x Points x Channel)'''
        # Project all points
        x = x.transpose(1, 2)
        x = self.point_proj(x)
        x = x.transpose(1, 2)
        
        # Pool points into a single descriptor if requested
        if self.pooling == 'max': x, _ = torch.max(x, 1)
        
        # Project descriptor if requested
        m = self.mu_head(x) if self.mu_head else None
        v = self.var_head(x) if self.var_head else None

        if m and v: return ModelReturnType(mu=m, var=v)
        elif m: return m
        else: return x


class TNet(nn.Module):
    ''' A PointNetProjector that produces a transform matrix given cloud.
    Given cloud (B, N, D) it returns a set of transforms (B, D, D)'''
    def __init__(self, dim:int, latent_dim:Optional[int]=None):
        super().__init__()
        self.dim = dim
        if latent_dim is None: latent_dim = dim
        self.proj = PointProjector([dim, latent_dim], variational=False)
        self.t_generator = LinearHead([dim, dim*dim])

    def forward(self, x):
        '''x: Batch x Points x Channel '''
        B = x.shape[0]
        x = self.proj(x)
        x = self.t_generator(x.squeeze(1)).view(B, self.dim, self.dim)
        return x
        

class PointNetEncoder(nn.Module):
    '''TODO: rename to PointNet.'''
    def __init__(self, dims:List[int]):
        super().__init__()
        self.dims = dims
        self.tnets = nn.ModuleList([TNet(d) for d in dims[:-1]])
        def get_projectors(dims):
            for i in range(len(dims)-2):
                yield PointProjector(dims[i:i+2], False, pooling=None)
            yield PointProjector(dims[-2:], False, pooling='max')
        self.proj = nn.ModuleList([m for m in get_projectors(dims)])

    def encoder(self):
        return self                    
    
    def forward(self, x, **_):
        "Expects a cloud of shape (Batch x Channel x Points)"
        for tnet, pnet in zip(self.tnets, self.proj):
            t = tnet(x)
            x = torch.einsum('bcd,bnd->bnc', t, x)
            x = pnet(x)
        return x.unsqueeze(1)