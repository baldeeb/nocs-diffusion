from utils.diffusion import (VarianceSchedule, Diffuser)
from utils.nocs_renderer import NOCSRenderer, sample_transforms
from utils.visualization import viz_image_batch
from utils.nocs_generator import PCloudLoader
import torch
from math import ceil

# Variance Scheduler Configs
num_steps   = 100
beta_1      = 1e-4
beta_max    = 0.01
show_count  = 9

# Stup Dataset
dataloader = PCloudLoader(path='./data/shapenet.hdf5',
                         categories=['chair'],
                         split='test',
                         batch_size=1,
                         shuffle=False,
                         device='cuda', )

# Setup Forward Diffusion
var_sched = VarianceSchedule(beta_1, beta_max, num_steps)
diffuse = Diffuser(var_sched)

# Setup Renderer
render = NOCSRenderer()

# Run Rendering
pts = dataloader()
Rs, Ts = sample_transforms(1)
diffed_bar = torch.stack([diffuse(pts[0], i) for i in range(0, num_steps, ceil(num_steps / show_count))])
images = torch.concatenate([render(pts[0], diffed_bar[i], Rs=Rs, Ts=Ts) for i in range(show_count)])
viz_image_batch(images[:,:,:,3:].detach().cpu().numpy())