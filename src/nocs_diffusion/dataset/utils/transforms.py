import torch
from pytorch3d.renderer import look_at_view_transform

def add_gaussian_noise(x, mu=0, std=0.005):
    return x + torch.randn(x.shape, device=x.device) * std + mu


def mask_from_depth(depth_ims, inverse=False):
    if inverse:
        masks = torch.ones_like(depth_ims)
        masks[depth_ims > 0] = 0
    else:
        masks = torch.zeros_like(depth_ims)
        masks[depth_ims > 0] = 1
    return masks

def rands_in_range(range, count):
    rand = torch.rand(count)
    return rand * (range[1] - range[0]) + range[0]


def sample_transforms(num_vars, 
                      dist_range=[0.6, 1.0],  
                      elev_range=[0, 360],  
                      azim_range=[0, 360],
                      device='cuda'):
    Rs, Ts = look_at_view_transform(
        rands_in_range(dist_range, num_vars),
        rands_in_range(elev_range, num_vars),
        rands_in_range(azim_range, num_vars),
        device=device
    )
    return Rs, Ts

class RandSe3NoiseFunctor:
    def __init__(self, 
                dist_range = [-0.2, 0.2],
                elev_range = [0,    70],
                azim_range = [0,    360],):
        self.dist_range = dist_range
        self.elev_range = elev_range
        self.azim_range = azim_range
    
    def __call__(self, d):
        Rs, Ts = sample_transforms(d.shape[0],
                                   self.dist_range,
                                   self.elev_range,
                                   self.azim_range,
                                    device=d.device)
        return torch.einsum('ijk,ilk->ijl',d, Rs) + Ts[:, None]
    

def nocs_extractor(x:torch.Tensor):
    '''
    Args:
        x: [B, W, H, 3]
    '''
    xmin = x.view(-1, x.shape[-1]).min(0).values
    xmax = x.view(-1, x.shape[-1]).max(0).values
    return (x / (xmax - xmin).norm()) + 0.5