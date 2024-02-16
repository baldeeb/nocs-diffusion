from utils.diffusion import (VarianceSchedule, Diffuser)
from utils.nocs_renderer import RendererWrapper, sample_transforms
from utils.visualization import viz_image_batch
from utils.dataloader import PointCloudLoader
import torch
from math import ceil

# Variance Scheduler Configs
num_steps   = 100
beta_1      = 1e-4
beta_max    = 0.01
show_count  = 9

# Stup Dataset
dataloader = PointCloudLoader(path='./data/shapenet.hdf5',
                         categories=['chair'],
                         split='test',
                         batch_size=1,
                         shuffle=False,
                         device='cuda', )

# Setup Forward Diffusion
var_sched = VarianceSchedule(beta_1, beta_max, num_steps)
diffuse = Diffuser(var_sched, mean=0.5)

# Setup Renderer
render = RendererWrapper()

# Run Rendering
pts = dataloader()
Rs, Ts = sample_transforms(1)
diffed_bar = torch.stack([diffuse(pts[0], i) for i in range(0, num_steps, ceil(num_steps / show_count))])
renders = [render(pts[0], diffed_bar[i], Rs=Rs, Ts=Ts) for i in range(show_count)]
images = torch.concatenate([r['images'] for r in renders])
images = images.clip(0.0, 1.0)
viz_image_batch(images.detach().cpu().numpy())

print('Done.')