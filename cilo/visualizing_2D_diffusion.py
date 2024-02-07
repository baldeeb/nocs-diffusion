from utils.diffusion import (VarianceSchedule, Diffuser)

from utils.visualization import viz_image_batch
from PIL import Image

import numpy as np
import torch
from utils.nocs_generator import NOCSObjectRenderer
from utils.nocs_generator import PointCloudLoader
from math import ceil

# Add noise
num_renders  = 6
steps        = 100
beta_1       = 1e-4
beta_max     = 0.025

# Setup Forward Diffusion
var_sched = VarianceSchedule(beta_1, beta_max, steps)
diffuse = Diffuser(var_sched, mean=0.5)

# Stup Dataset
dataloader = PointCloudLoader(path='./data/shapenet.hdf5',
                         categories=['chair'],
                         split='test',
                         batch_size=1,
                         shuffle=False,
                         device='cuda', )
render = NOCSObjectRenderer(dataloader)


renders = render()
image = renders['images'][0]
depth = renders['depths'][0]

mask = torch.zeros_like(depth)
mask[depth!=-1] = 1
mask = mask.permute(1,2,0)[None]

diff_ims = torch.stack([diffuse(image, i) for i in range(0, steps, ceil(steps/num_renders))])
diff_ims = diff_ims.clip(0.0, 1.0)

if mask is not None: diff_ims *= mask

viz_image_batch(diff_ims.detach().cpu().numpy())
print("Done.")