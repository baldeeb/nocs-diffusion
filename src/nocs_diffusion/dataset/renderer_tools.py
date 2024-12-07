import torch
from pytorch3d.renderer import look_at_view_transform


# def sample_from_clouds(data, out_count, pad_zeros=True):
#     """ 
#     Point clouds can have unequal number of points. 
#     This functions facilitates making those clouds of equal size.
#     """
#     for i in range(len(data)):
#         pt_count = len(data[i]) 
#         if pt_count < out_count:
#             if pad_zeros:               # Pad with zeros
#                 count_diff = out_count - pt_count
#                 data[i].extend(torch.zeros(count_diff, 3))
#             else:                       # Oversample
#                 pt_idxs = torch.multinomial(torch.ones(out_count), 
#                                             out_count, 
#                                             replacement=True)
#                 data[i] = data[i][pt_idxs]
#         elif pt_count > out_count:      # Undersample
#             data[i] = data[i][torch.randperm(out_count)]
        
#     return torch.stack(data)

def sample_from_clouds(data, out_count, pad_zeros=True):
    B, N, D = data.shape
    if N < out_count:
        if pad_zeros:               # Pad with zeros
            raise RuntimeError
        else:
            raise RuntimeError
    elif N > out_count:
        pi = ((torch.rand(B, out_count) - 1e-6) * N).flatten().long()
        bi = (torch.ones(B, out_count) * torch.arange(B)[:, None]).flatten().long()
        data = data[bi, pi].reshape(B, out_count, D)
        
    return data


def add_gaussian_noise(x, mu=0, std=0.005):
    return x + torch.randn(x.shape, device=x.device) * std + mu


def mask_from_depth(depth_ims, inverse=False):
    if inverse:
        masks = torch.ones_like(depth_ims)
        masks[depth_ims!=-1] = 0
    else:
        masks = torch.zeros_like(depth_ims)
        masks[depth_ims!=-1] = 1
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

class RandSe3Noise:
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