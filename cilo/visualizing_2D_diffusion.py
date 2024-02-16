from utils.diffusion import (VarianceSchedule, Diffuser)
from utils.visualization import viz_image_batch
from utils.nocs_generator import nocs_extractor
from utils.dataloader import PointCloudLoader
from utils.nocs_renderer import RendererWrapper, mask_from_depth

from math import ceil
from time import time

import torch


batch_size   = 1

# Add noise
num_renders  = 6
steps        = 100
beta_1       = 1e-4
beta_max     = 0.01

# Setup Forward Diffusion
var_sched = VarianceSchedule(beta_1, beta_max, steps)
diffuse = Diffuser(var_sched, mean=0.5)

def add_noise(x, mu=0, std=0.005):
    return x + torch.randn(x.shape, device=x.device) * std + mu

# Stup Dataset
dataloader = PointCloudLoader(path='./data/shapenet.hdf5',
                         categories=['chair'],
                         split='test',
                         batch_size=batch_size,
                         shuffle=False,
                         device='cuda', )
render = RendererWrapper()

t0 = time()
print(f'Started @ {t0}')

x  = dataloader()
fx = nocs_extractor(x)

# TEMP
x = x[0]
fx = fx[0]


diff_fx = torch.stack([diffuse(fx, i) for i in range(0, steps, ceil(steps/num_renders))])
diff_x = torch.stack([add_noise(x) for i in range(0, steps, ceil(steps/num_renders))])
renders = render(diff_x, diff_fx, 1)

# render = NOCSObjectRenderer(dataloader)
# renders = render()
images = renders['images']
depths = renders['depths']

mask = mask_from_depth(depths)
# mask = mask.permute(1,2,0)[None]

# diff_ims = torch.stack([diffuse(image, i) for i in range(0, steps, ceil(steps/num_renders))])
# diff_ims = diff_ims.clip(0.0, 1.0)
diff_ims = images.clip(0.0, 1.0)

if mask is not None: diff_ims *= mask
print(f'Done processing after {time() - t0}')

viz_image_batch(diff_ims.detach().cpu().numpy())
print("Visualization complete!")