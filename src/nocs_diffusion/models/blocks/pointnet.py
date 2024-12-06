import torch
from torch import nn
from .linear_blocks import SetLinearHead, LinearHead 
from ..utils import ModelReturnType
from typing import List, Optional

class PointNetProjector(nn.Module):
    def __init__(self, in_dim:int, layer_dims:List, out_dim:int, variational:bool=True, pooling:str="max"):
        super().__init__()
        self.dims = [in_dim] + layer_dims + [out_dim] 
        self.pooling = pooling
        
        if len(self.dims) <= 2: 
            self.point_proj = nn.Identity()
        else:
            SetLinearHead(self.dims[:-1], post_bn=True)
        self.mu_head    = LinearHead(self.dims[-2:])
        self.var_head   = LinearHead(self.dims[-2:]) if variational else lambda _: torch.FloatTensor()

    def forward(self, clouds):
        x = clouds
        x = x.transpose(1, 2)
        x = self.point_proj(x)
        x = x.transpose(1, 2)
        if self.pooling == 'max': x, _ = torch.max(x, 1)
        m = self.mu_head(x)
        v = self.var_head(x)

        # Returns both mean and logvariance, just ignore the latter in deteministic cases.
        return ModelReturnType(mu=m, var=v)



class PoinNetTNet(nn.Module):
    def __init__(self, dim:int, latent_dim:Optional[int]=None):
        super().__init__()
        self.dim = dim
        if latent_dim is None: latent_dim = dim
        self.proj = PointNetProjector(dim, [], latent_dim, variational=False)
        self.t_generator = LinearHead([dim, dim*dim])

    def forward(self, x):
        '''x: Batch x Points x Channel '''
        B = x.shape[0]
        x = self.proj(x).mu
        x = self.t_generator(x.squeeze(1)).view(B, self.dim, self.dim)
        return x
        

class PointNetEncoder(nn.Module):
    def __init__(self, dims:List[int]):
        super().__init__()
        self.dims = dims
        self.tnets = nn.ModuleList([PoinNetTNet(d) for d in dims[:-1]])
        def get_projectors(dims):
            for i, o in zip(dims[:-2], dims[1:-1]):
                yield PointNetProjector(i, [], o, False, pooling=None)
            yield PointNetProjector(dims[-2], [], dims[-1], False, pooling='max')
        self.proj = nn.ModuleList([m for m in get_projectors(dims)])

    def encoder(self):
        return self                    
    
    def forward(self, cloud, **_):
        x = cloud
        for tnet, pnet in zip(self.tnets, self.proj):
            t = tnet(x)
            x = torch.einsum('bcd,bnd->bnc', t, x)
            x = pnet(x).mu
        return x.unsqueeze(1)