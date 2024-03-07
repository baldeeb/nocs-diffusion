import torch
from pytorch3d.renderer import look_at_view_transform


def sample_from_clouds(data, out_count, pad_zeros=True):
    """ 
    Point clouds can have unequal number of points. 
    This functions facilitates making those clouds of equal size.
    """
    for i in range(len(data)):
        pt_count = len(data[i]) 
        if pt_count < out_count:
            if pad_zeros:               # Pad with zeros
                count_diff = out_count - pt_count
                data[i].extend(torch.zeros(count_diff, 3))
            else:                       # Oversample
                pt_idxs = torch.multinomial(torch.ones(out_count), 
                                            out_count, 
                                            replacement=True)
                data[i] = data[i][pt_idxs]
        elif pt_count > out_count:      # Undersample
            data[i] = data[i][torch.randperm(out_count)]
        
    return torch.stack(data)


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