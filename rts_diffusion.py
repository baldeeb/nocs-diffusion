'''
Say we have a depth image and ground truth object shape
can we retrieve rotation, translation, and scale of that
object from the observed points
'''

from utils.dataset import *
from utils.misc import *
from utils.data import *

from utils.data_augmentation import CollateFunctor

import torch
from torch import nn

import wandb

from models.encoders.pointnet import PointNetEncoder
from models.pose_diffusion import PoseDiffModel
from models.common import *
from pytorch3d.transforms import quaternion_to_matrix as quat2mat
from pytorch3d.transforms import random_quaternions as rand_quat

from utils.datamanip import (
    apply_transform, 
    get_random_qts,
)



device = 'cuda:0'

# Datasets and loaders
test_dset = ShapeNetCore(
    path='./data/shapenet.hdf5',
    cates=['chair'],
    split='test',
    scale_mode='shape_unit'
)

test_loader = get_data_iterator(DataLoader(test_dset, batch_size=8, num_workers=0, shuffle=False))

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

# subsets = try_halfspace_split(in_points, 5)
# print(f'we are left with {subsets.shape}')
# visualize_point_cloud(subsets[0].detach().cpu().numpy())
# print(f"out of {in_points.shape[N_ax]} points, {subsets.shape[N_ax]} points are in the depth image")



def evaluate(model:PoseDiffModel, points):
    '''
    Args:
        model (nn.Module): 
        data (torch.Tensor): (B, N, D) tensor of points
    '''
    model.eval()
    B = points.shape[0]
    dismantle = lambda qts: (qts[:, 0, :4], qts[:, 0, 4:7], qts[:, 0, 7:8])
    with torch.no_grad():
        q, t, s = get_random_qts(B)
        
        pts_T = apply_transform(points, q, t, s)
        z_mu, _ = model.encoder(pts_T.to(device))

        qts_0 = get_random_qts(B, catted=True).unsqueeze(1)
        qts_hat = model.diffusion(qts_0, z_mu)
        # pts_hat = apply_transform(points, *dismantle(qts_hat))
        
        # calculate loss for quaternion, translation, and scale
        q_hat, t_hat, s_hat = dismantle(qts_hat)
        loss_q = (q.to(q_hat) - q_hat).abs().mean()
        loss_t = (t.to(t_hat) - t_hat).abs().mean()
        loss_s = (s.to(s_hat) - s_hat).abs().mean()
        
        # visualize_point_cloud(x_T[0].detach().cpu().numpy(), colors=(1,0,0), show=False)
        # visualize_point_cloud(x_T_hat[0].detach().cpu().numpy(), colors=(0,1,0))
        return {'loss_q': loss_q, 'loss_t': loss_t, 'loss_s': loss_s}
        

import time
import pathlib as pl
class ModelSaver():
    def __init__(self, folder, sufix, period=100):
        self.PERIOD = period
        self.SUFIX = sufix
        self.DIR = pl.Path(f'./{folder}/{time.time()}')
        self.DIR.mkdir(parents=True, exist_ok=True)
    def __call__(self, model, epoch):
        if epoch % self.PERIOD != 0: return
        FILE = pl.Path(f'{epoch}_{self.SUFIX}.pt')
        torch.save(model.state_dict(), self.DIR/FILE)

def train(model:nn.Module, 
          dataloader:DataLoader,
          learning_rate:float=1e-3,
          num_epochs:int=10000, 
          log:str=lambda *_:None, 
          save:ModelSaver=lambda *_:None):
    optimizer = torch.optim.Adam(model.parameters(), 
                                 lr=learning_rate)
    model.train()
    for epoch in tqdm(range(num_epochs)):
        log({'epoch': epoch})
        for points in dataloader:
            B, _, _ = points.shape  # Update batch size
            x = torch.as_tensor(points).to(device)
            q, t, s = get_random_qts(B)
            x_t = apply_transform(x, q, t, s)

            loss = model.get_loss(x_t, torch.cat([q, t, s], dim=1))
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            log({'loss': loss.item()})
            save(model, epoch)
            eval_losses = evaluate(model, points[:100])
            log(eval_losses)

n_augs = 8
batch_size = 32


name=f'category-level-batch{batch_size}x{n_augs}'
model_saver = ModelSaver('checkpoints', name, 
                         period=1000)
collate_fn = CollateFunctor(num_aug=n_augs)
dataloader = DataLoader(test_dset, 
                        collate_fn=collate_fn, 
                        batch_size=batch_size, 
                        num_workers=0, 
                        shuffle=True)
model = PoseDiffModel().to(device)
wandb.init(project="diffusion", name=name)
train(model, dataloader, log=wandb.log, save=model_saver)
# train(model, dataloader)  # For debugging

# subsets = try_halfspace_split(in_points, 10)
# print(evaluate(model, subsets))

