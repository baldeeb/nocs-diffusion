
from torch.nn.functional import normalize
from torch import where, rand, randperm, stack
import torch
from utils.dataset import *
from utils.misc import *
from utils.data import *



def halfspace_split(in_points, desired_batch, min_pts=250, max_pts=1000):
    '''
    Args:   
        in_points (torch.Tensor): (num_points, 3). The device on which
            the points are stored is used to generate the random halfplane
        out_batch (int): The number of augmentations to generate
        min_pts (int): The minimum number of points in the halfspace
        max_pts (int): The maximum number of points in the halfspace
    rasies:
        RuntimeError: If the number of points in the halfspace is less than min_pts
    '''
    
    device = in_points.device
    N = in_points.shape[0]

    # Select a random halfplane
    hp_normals = normalize(torch.rand(3, desired_batch) - 0.5).to(device) # 1 x B
    hp_offsets = (rand(1, desired_batch).to(device) * 2 - 1) * 0.002      # 1 x B
    hp_dots = in_points @ hp_normals                                  # N x B
    
    # Select points in the halfplane
    select = (hp_dots > hp_offsets).squeeze()                         # N x B
    sample_counts = select.sum(axis=0)
    # num_samples = min(sample_counts.min(), max_pts)
    # if num_samples < min_pts: raise RuntimeError('Not enough points in the halfspace')
    batch_idxs = torch.arange(desired_batch, dtype=torch.int32, device=device)[sample_counts > min_pts]
    select = select[:, batch_idxs]
    num_samples = min(select.sum(axis=0).min(), max_pts)
    subsets = []
    for b_idx in range(len(batch_idxs)):
        i_pts = where(select[:, b_idx])[0]  # Find points in +ive halfspace
        shuffled = randperm(len(i_pts), dtype=torch.int32, device=device)
        subsets.append(in_points[i_pts[shuffled][:num_samples], :]) 
    return stack(subsets, axis=0)

def try_halfspace_split(in_points, batch, min_pts=250, max_pts=1000, tries=10):
    '''
    Tries to run halfspace_split() for a given number of times
    
    '''
    subsets = None
    for _ in range(tries):
        try:
            subsets = halfspace_split(in_points, batch, min_pts, max_pts)
            return subsets
        except RuntimeError: 
            pass
    raise RuntimeError('Could not find a good split')



class CollateFunctor:
    def __init__(self, num_aug=256, min_pts=200, max_pts=1000, tries=1000):
        self.n_aug = num_aug
        self.min_p = min_pts
        self.max_p = max_pts
        self.tries = tries
    
    def _get_auged(self, points):
        for p in points:
            yield try_halfspace_split(p, self.n_aug, 
                                      self.min_p, self.max_p, 
                                      tries=self.tries) 
            
    def __call__(self, batch):
        pointcloud = torch.stack([item['pointcloud'] for item in batch])
        shift = torch.stack([item['shift'] for item in batch])
        scale = torch.stack([item['scale'] for item in batch])
        points = (pointcloud * scale) + shift
        
        data = list(self._get_auged(points))
        N = min([s.shape[1] for s in data])
        return torch.concat([s[:, :N] for s in data], dim=0)
