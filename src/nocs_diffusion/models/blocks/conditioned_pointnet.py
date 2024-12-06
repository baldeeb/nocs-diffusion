import torch
from torch import nn

from .film import FilmLinearLayer
from ..utils import ModelReturnType

from typing import List
    
class ConditionedPointNetEncoder(nn.Module):
    def __init__(self, dims:List, ctxt_dim:int,
                 last_proj_layers=None,
                  variational=False ):
        super().__init__()
        self.dims = dims
        film_ch = dims[:-1]
        film_out = last_proj_layers[0] if last_proj_layers else dims[-1]
        self.enc_layers = []
        for i, j in zip(film_ch[:-1], film_ch[1:]):
            self.enc_layers.append(FilmLinearLayer(i, j, 
                                                   ctxt_ch_in=ctxt_dim,
                                                   kernel_size=1))
        self.enc_layers.append(FilmLinearLayer(film_ch[-1], film_out, 
                                               ctxt_ch_in=ctxt_dim,
                                               act_layer=nn.Identity, 
                                               kernel_size=1))
        self.enc_layers = nn.ModuleList(self.enc_layers)
        
        def build_proj(proj_dims: List):
            def _layer(i, o, a=True): 
                l = [nn.Conv1d(i, o, 1), nn.BatchNorm1d(o)]    
                if a: l.append(nn.ReLU())
                return nn.Sequential(*l)
            for i, j in zip(proj_dims[:-2], proj_dims[1:-1]):
                yield _layer(i, j)
            yield _layer(proj_dims[-2], proj_dims[-1], False)
        
        proj_ch = last_proj_layers + [dims[-1]] if last_proj_layers else [dims[-1], dims[-1]]
        self.mu = nn.Sequential(*(m for m in build_proj(proj_ch))) 
        self.var = nn.Sequential(*(m for m in build_proj(proj_ch))) if variational else None

    def forward(self, clouds, ids=None, **_):
        x = clouds
        # Batch, Points, Channel = x.shape
        x = x.transpose(1, 2)
        for l in self.enc_layers:
            x = l(x, ctxt=ids)
        
        if self.var:
            out = ModelReturnType(mu = self.mu(x).transpose(1, 2),
                                  var = self.var(x).transpose(1, 2))
        else: 
            out = self.mu(x).transpose(1, 2)
        return out
