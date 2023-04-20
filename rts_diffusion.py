'''
Say we have a depth image and ground truth object shape
can we retrieve rotation, translation, and scale of that
object from the observed points
'''
import numpy as np

from torch.nn.functional import normalize

from utils.dataset import *
from utils.misc import *
from utils.data import *

from utils.datamanip import (
    project_point_cloud,
    visualize_point_cloud,
    random_SE3,
)

import matplotlib.pyplot as plt

device = 'cuda'

# Datasets and loaders
test_dset = ShapeNetCore(
    path='./data/shapenet.hdf5',
    cates=['chair'],
    split='test',
    scale_mode='shape_unit'
)
test_loader = get_data_iterator(DataLoader(test_dset, batch_size=1, num_workers=0, shuffle=False))

print(f'Loading data...')
batch = next(test_loader)
ref = batch['pointcloud'].to(device)
shift = batch['shift'].to(device)
scale = batch['scale'].to(device)
in_points = (ref * scale) + shift

N_ax, dim_ax = 0, -1  # Those are the axis for num_points and points_dim
B, N, D = in_points.shape
# visualize_point_cloud(in_points[0].detach().cpu().numpy())

# print(f'lets see what we are working with:')
# print(f'min {in_points.min(axis=1).values}')
# print(f'max: {in_points.max(axis=1).values}')
# print(f'shift: {shift}, scale: {scale}')
# print(f'inpoints shape: {in_points.shape}\n')

def halfspace_split(in_points, out_batch):
    '''
    Args:   
        in_points:  num_points x 3
    '''
    device = in_points.device
    N = in_points.shape[0]

    # Select a random halfplane
    hp_normals = normalize(torch.rand(3, out_batch) - 0.5).to(device)  # (x, y, z) x batch
    hp_offsets =  torch.rand(1, out_batch).to(device) * 0.002           # (offset) x batch
    # hp_offsets =  torch.zeros(1, out_batch).to(device)           # (offset) x batch
    # print(f'Splitting the points using plane {hp_normals} and offset {hp_offsets}')

    # Which points are on the positive side of the plane?
    hp_dots = in_points @ hp_normals  # num_points x batch

    subsets = []
    min_pts, max_pts = 250, 1000
    select = (hp_dots > hp_offsets).squeeze()  # num_points x batch
    num_samples = min(select.sum(axis=0).min(), max_pts)
    if num_samples < min_pts: raise RuntimeError('Not enough points in the halfspace')
    for b_idx in range(out_batch):
        i_pts = torch.where(select[:, b_idx])[0]  # Find points in +ive halfspace
        # i_chosen = np.random.choice(i_pts, num_samples, replace=False)
        shuffled = torch.randperm(len(i_pts), dtype=torch.int32, device=device)
        subsets.append(in_points[i_pts[shuffled][:num_samples], :]) 
    return torch.stack(subsets, axis=0)

def try_halfspace_split(in_points, batch, tries=10):
    subsets = None
    for _ in range(tries):
        try:
            subsets = halfspace_split(in_points[0], batch)
            return subsets
        except RuntimeError: pass
    raise RuntimeError('Could not find a good split')

# subsets = try_halfspace_split(in_points, 5)
# print(f'we are left with {subsets.shape}')
# visualize_point_cloud(subsets[0].detach().cpu().numpy())
# print(f"out of {in_points.shape[N_ax]} points, {subsets.shape[N_ax]} points are in the depth image")



import torch
from torch import nn

import wandb

from models.encoders.pointnet import PointNetEncoder
from models.diffusion import DiffusionPoint, VarianceSchedule, PointwiseNet, ContextualDiffusion
from models.common import *
from pytorch3d.transforms import quaternion_to_matrix as quat2mat
from pytorch3d.transforms import random_quaternions as rand_quat

class DiffusionModel(nn.Module):
    def __init__(self):
        super().__init__()
        pt_dim = 3
        T_dim = 4 + 3 + 1  # quaternion + translation + scale
        num_steps = 100
        beta_1 = 1e-4
        beta_T = 0.02
        sched_mode = 'linear'
        latent_dim = 256
        residual = True

        self.encoder = PointNetEncoder(
            latent_dim, 
            pt_dim
        )
        self.diffusion = ContextualDiffusion(
            net = PointwiseNet(
                point_dim=T_dim, 
                context_dim=latent_dim, 
                residual=residual
            ),
            var_sched = VarianceSchedule(
                num_steps=num_steps,
                beta_1=beta_1,
                beta_T=beta_T,
                mode=sched_mode,
            )
        )

    def __call__(self, points, transform=None):
        if self.training():
            assert transform is not None, 'Must provide a transform in training mode'
            return self.get_loss(points, transform)
        else:
            return self.get_transform(points)

    def get_transform(self, points):
        '''
        Args:
            points: (B, N, D) tensor of context points 
                (typcially D=3)
        '''
        points = points.to(self.device)
        z_mu, _ = self.encoder(points)
        transform = self.diffusion.get_transform(z_mu)
        return transform

    def to(self, device):
        self.device = device
        return super().to(device)

    def get_loss(self, points, transform):
        '''
        Args:
            context: (B, N, D) tensor of context points 
                (typcially D=3)
            x: (B, 4+3+1) tensor of query SE3 transform
        '''
        # Question: Shuold the context be diffused?
        # Answer: not atm. It will stop being a normal diffusion model if the 
        #       context is changed with the diffusion.

        points = points.to(self.device)
        transform = transform.unsqueeze(1).to(self.device)
        z_mu, _ = self.encoder(points)
        loss_recons = self.diffusion.get_loss(transform, z_mu)

        return loss_recons

def apply_transform(x, quat, t, s):
    '''
    Args:
        x: (B, N, D) tensor of points
        quat: (B, 4) tensor of quaternions
        t: (B, 3) tensor of translations
        s: (B, 1) tensor of scales
    '''
    T = quat2mat(quat).to(x)
    puff_up = lambda v: v.unsqueeze(1).to(x)
    s, t = puff_up(s), puff_up(t)
    return (x @ T * s) + t


def get_random_qts(B=1):
    quat = rand_quat(B)
    t = torch.rand(B, 3)*0.25
    s = torch.rand(B, 1)*0.25 + 1.0
    return quat, t, s


def evaluate(model:DiffusionModel, points):
    '''
    Args:
        model: DiffusionModel
        data: (B, N, D) tensor of points
    '''
    model.eval()
    B = points.shape[0]
    dismantle = lambda qts: (qts[:, 0, :4], qts[:, 0, 4:7], qts[:, 0, 7:8])
    with torch.no_grad():
        q, t, s = get_random_qts(B)
        pts_T = apply_transform(points, q, t, s)
        qts_0 = torch.cat([q, t, s], dim=1).unsqueeze(1)
        z_mu, _ = model.encoder(pts_T)
        qts = model.diffusion(qts_0, z_mu)
        pts_hat = apply_transform(points, *dismantle(qts))
        
        # calculate loss for quaternion, translation, and scale
        q_hat, t_hat, s_hat = dismantle(qts)
        loss_q = (q.to(q_hat) - q_hat).abs().mean()
        loss_t = (t.to(t_hat) - t_hat).abs().mean()
        loss_s = (s.to(s_hat) - s_hat).abs().mean()
        
        # visualize_point_cloud(x_T[0].detach().cpu().numpy(), colors=(1,0,0), show=False)
        # visualize_point_cloud(x_T_hat[0].detach().cpu().numpy(), colors=(0,1,0))
        return {'loss_q': loss_q, 'loss_t': loss_t, 'loss_s': loss_s}
        

import time
import pathlib as pl
class ModelSaver():
    def __init__(self, folder, sufix):
        self.SUFIX = sufix
        self.DIR = pl.Path(f'./{folder}/{time.time()}')
        self.DIR.mkdir(parents=True, exist_ok=True)
    def __call__(self, model, epoch):
        FILE = pl.Path(f'{epoch}_{self.SUFIX}.pt')
        torch.save(model.state_dict(), self.DIR/FILE)

save = ModelSaver('checkpoints', 'diffusion')
model = DiffusionModel().to(device)

def train(model:DiffusionModel, in_points):

    LOG = True
    if LOG: wandb.init(project="diffusion", name="mem-test")
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for epoch in tqdm(range(10000)):
        if LOG: wandb.log({'epoch': epoch})

        try: subsets = try_halfspace_split(in_points, 1024)
        except RuntimeError:
            if LOG: wandb.log({'batch_discarded': 1}) 
            continue
        if LOG: wandb.log({'batch_discarded': 0})
        
        B, _, _ = subsets.shape  # Update batch size
        x = torch.as_tensor(subsets).to(device)
        q, t, s = get_random_qts(B)
        x_t = apply_transform(x, q, t, s)

        loss = model.get_loss(x_t, torch.cat([q, t, s], dim=1))
        
        optim.zero_grad(); loss.backward(); optim.step()

        if LOG: wandb.log({'loss': loss.item()})

        if epoch % 1000 == 0: save(model, epoch)
        eval_losses = evaluate(model, subsets[:100])
        if LOG: wandb.log(eval_losses)



train(model, in_points)

# subsets = try_halfspace_split(in_points, 10)
# print(evaluate(model, subsets))






# print(f'we got transform \n{T_pred}')

# pred_subset = T_pred[:, :, 0:3] @ x.transpose(1, 2) + T_pred[:, :, 3:4]
# print(pred_subset.shape)

# print(f'now we need to try and train this to predict')
