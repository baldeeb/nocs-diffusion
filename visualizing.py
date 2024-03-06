from utils.diffusion import (VarianceSchedule, ForwardDiffuser)
from utils.renderer import RendererWrapper, sample_transforms, mask_from_depth
from utils.visualization import viz_image_batch
from utils.dataloader import PointCloudLoader
from utils.nocs_generator import nocs_extractor
import torch
from math import ceil

def add_noise(x, mu=0, std=0.005):
    return x + torch.randn(x.shape, device=x.device) * std + mu

######## Configs ############
batch_size   = 1

# Variance Scheduler 
num_renders  = 25
steps        = 100
beta_1       = 1e-4
beta_max     = 0.005
#############################

# Stup Dataset
dataloader = PointCloudLoader(path='./data/shapenet.hdf5',
                         categories=['chair'],
                         split='test',
                         batch_size=1,
                         shuffle=False,
                         device='cuda', )

# Setup Forward Diffusion
var_sched = VarianceSchedule(beta_1, beta_max, steps)
diffuse = ForwardDiffuser(var_sched, mean=0.5)
diffuse_clip = lambda *args: diffuse(*args).clip(0.0, 1.0)

# Setup Renderer
render = RendererWrapper()

# Run Rendering
pts = dataloader()
feats = nocs_extractor(pts)


sample_i_gen = range(0, steps, ceil(steps/num_renders))
diff_feats = torch.stack([diffuse_clip(feats, i) for i in sample_i_gen])
diff_pts =   torch.stack([add_noise(pts)         for _ in sample_i_gen])


Rs, Ts = sample_transforms(1)
renders = [render(x, fx, Rs=Rs, Ts=Ts) for x, fx in zip(diff_pts, diff_feats)]

depths = torch.concatenate([r['depths'] for r in renders])
masks = mask_from_depth(depths)
images = torch.concatenate([r['images'] for r in renders])
images = images * masks

viz_image_batch(images.detach().cpu().numpy())
print("Visualization complete!")