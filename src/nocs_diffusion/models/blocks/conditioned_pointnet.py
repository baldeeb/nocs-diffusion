import torch
from torch import nn

from .film import FilmLinearLayer
from .pointnet import TNet
from ..utils import ModelReturnType

from typing import List
    
class ConditionedPointProjector(nn.Module):
    def __init__(self, dims:List, ctxt_dim:int, pooling:str='max',
                 last_proj_dims=None, variational=False ):
        super().__init__()

        if variational and not last_proj_dims: 
            raise RuntimeError("When using variational inference" +
                               "a final non-FiLM layer is required")
        self.dims = dims
        self.pooling = pooling
        
        def build_film_layers(dims):
            if len(dims) > 2:
                for i in range(len(dims) - 2):
                    yield FilmLinearLayer(dims[i], dims[i+1], ctxt_ch_in=ctxt_dim,
                                        kernel_size=1)
            yield FilmLinearLayer(dims[-2], dims[-1], ctxt_ch_in=ctxt_dim, 
                                act_layer=nn.Identity, kernel_size=1)
        
        def build_proj(proj_dims: List):
            def _layer(i, o, a=True): 
                l = [nn.Conv1d(i, o, 1), nn.BatchNorm1d(o)]    
                if a: l.append(nn.ReLU())
                return nn.Sequential(*l)
            for i, j in zip(proj_dims[:-2], proj_dims[1:-1]):
                yield _layer(i, j)
            yield _layer(proj_dims[-2], proj_dims[-1], False)
        
        self.enc_layers = nn.ModuleList((l for l in build_film_layers(dims)))
        self.mu = self.var = None
        if last_proj_dims:
            self.mu = nn.Sequential(*(m for m in build_proj(dims[-1:]+last_proj_dims))) 
            if variational:
                self.var = nn.Sequential(*(m for m in build_proj(dims[-1:]+last_proj_dims))) 

    def forward(self, x, ctxt=None, **_):
        # Batch, Points, Channel = x.shape
        x = x.transpose(1, 2)
        for l in self.enc_layers:
            x = l(x, ctxt=ctxt)  # x -> (B, C, P)
            
        if self.pooling == 'max': x, _ = torch.max(x, -1, keepdim=True)
        
        mu  = self.mu(x).transpose(1,2)  if self.mu else None   
        var = self.var(x).transpose(1,2) if self.var else None   

        if mu and var: out = ModelReturnType(mu=mu, var=var)
        elif mu:       out = mu
        else:          out = x.transpose(1, 2)
        
        return out


class ConditionedPointNetEncoder(nn.Module):

    def __init__(self, dims:List, ctxt_dim:int,
                  *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.tnets = nn.ModuleList([TNet(d) for d in dims[:-1]])
        def get_projectors(dims, ctxt_dim):
            for i in range(len(dims)-2):
                yield ConditionedPointProjector(dims[i:i+2], ctxt_dim, pooling=None)
            yield ConditionedPointProjector(dims[-2:], ctxt_dim, pooling='max')
        self.proj = nn.ModuleList([m for m in get_projectors(dims, ctxt_dim)])

    def encoder(self):
        return self                    
    
    def forward(self, x, ctxt=None, **_):
        "Expects a cloud of shape (Batch, Points, Channel)"
        for tnet, pnet in zip(self.tnets, self.proj):
            t = tnet(x)
            x = torch.einsum('bcd,bnd->bnc', t, x)
            x = pnet(x, ctxt)
        return x