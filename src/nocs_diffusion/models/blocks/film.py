import torch
from torch import nn

class FilmLinearLayer(nn.Module):
    def __init__(self, in_ch, out_ch, 
                 ctxt_ch_in=None,
                 bn_layer=nn.BatchNorm1d, 
                 act_layer=nn.ReLU, 
                 **conv_kwargs):
        self.conv =  nn.Conv1d(in_ch, out_ch, **conv_kwargs)
        self.bn = bn_layer(out_ch)
        self.act = act_layer()
        if ctxt_ch_in is not None:
            self.ctxt_proj = nn.Linear(ctxt_ch_in, 2 * out_ch)
        else: self.ctxt_proj = nn.Identity()

    def forward(self, x, ctxt=None):
        if ctxt is not None:
            alpha, beta = torch.chunk(self.ctxt_proj(ctxt), 2)
        else: alpha, beta = 1.0, 0.0
        return self.act(self.bn( self.conv(x) * alpha + beta ))

class FilmResLayer(nn.Module):
    def __init__(self, in_dim, out_dim, ctx_dim):
        super().__init__()

        self.conv1 = nn.Conv2d(in_dim, out_dim, 3, 1, 1)
        self.activ1 = nn.GELU()

        
        self.conv2 = nn.Conv2d(out_dim, out_dim, 1)
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
