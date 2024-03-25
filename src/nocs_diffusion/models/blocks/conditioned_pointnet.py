from torch import nn

from .film import FilmLinearLayer
from ..utils import ModelReturnType
    
class ConditionedPointNetEncoder(nn.Module):
    def __init__(self, zdim, in_ch=3,
                 film_ch=[128, 128, 256, 512],
                 proj_ch=[512, 256, 128]
            ):
        film_ch = [in_ch] + film_ch
        self.enc_layers = []
        for i, j in zip(film_ch[:-2], film_ch[1:-1]):
            self.enc_layers.append(FilmLinearLayer(i, j, kernel_size=1))
        self.enc_layers.append(FilmLinearLayer(film_ch[-2], film_ch[-1], 
                                               act_layer=nn.Identity, 
                                               kernel_size=1))
        def _layer(i, o, a=True): 
            l = [nn.Linear(i, o), nn.BatchNorm1d(o)]    
            if a: l.append(nn.ReLU())
            return nn.Sequential(*l)
        
        proj_ch = proj_ch + [zdim]
        def build_proj():
            for i, j in zip(proj_ch[:-2], proj_ch[1:-1]):
                yield _layer(i, j)
            yield _layer(proj_ch[-2], film_ch[-1], False)
        self.mu = nn.Sequential(build_proj()), 
        self.var = nn.Sequential(build_proj())

    def forward(self, x, ctxt=None):
        for l in self.enc_layers:
            x = l(x, ctxt=ctxt)
        return ModelReturnType(mu = self.mu(x), var=self.var(x))
        # return self.mu(x), self.var(x)
