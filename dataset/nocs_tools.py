import torch

def nocs_extractor(x:torch.Tensor):
    '''
    Args:
        x: [B, W, H, 3]
    '''
    xmin = x.view(-1, x.shape[-1]).min(0).values
    xmax = x.view(-1, x.shape[-1]).max(0).values
    return (x / (xmax - xmin).norm()) + 0.5
