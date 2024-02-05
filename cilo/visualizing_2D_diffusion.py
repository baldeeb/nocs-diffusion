from utils.diffusion import (VarianceSchedule, Diffuser)

from utils.visualization import viz_image_batch
from PIL import Image

import numpy as np
import torch
from utils.nocs_generator import get_data_generator
from math import ceil

# Add noise
num_renders  = 6
steps   = 200
beta_1      = 1e-4
beta_max    = 0.025

# Setup Forward Diffusion
var_sched = VarianceSchedule(beta_1, beta_max, steps)
diffuse = Diffuser(var_sched)

gen = get_data_generator(
    path='./data/shapenet.hdf5',
    cates=['chair'],
    split='test',
    shuffle=False,
    device='cuda', 
    batch_size=1
)

data = gen()
image = data[0, :, :, :3]
mask = torch.zeros_like(image)
mask[image > -0.5] = 1
mask = (mask[:, :, 0].bool() | mask[:, :, 1].bool() | mask[:, :, 2].bool())[:, :, None].float()  

diff_ims = torch.stack([diffuse(image, i) for i in range(0, steps, ceil(steps/num_renders))])
diff_ims = diff_ims.clip(-0.5, 0.5) + 0.5

if mask is not None:
    diff_ims *= mask

viz_image_batch(diff_ims.detach().cpu().numpy())