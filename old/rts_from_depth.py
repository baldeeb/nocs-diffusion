'''
Say we have a depth image and ground truth object shape
can we retrieve rotation, translation, and scale of that
object from the observed points
'''
import numpy as np

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
test_loader = get_data_iterator(DataLoader(test_dset, batch_size=1, num_workers=0))

print(f'Loading data...')
batch = next(test_loader)
ref = batch['pointcloud'].to(device)
shift = batch['shift'].to(device)
scale = batch['scale'].to(device)

in_points = (ref * scale) + shift

print(f'lets see what we are working with:')
print(f'min {in_points.min(axis=1).values}, max: {in_points.max(axis=1).values}')
print(f'shift: {shift}, scale: {scale}')
print('\n')

# Visualize transformed pointcloud
# visualize_point_cloud(in_points[0].detach().cpu().numpy())

# Generate a random SE(3) transformation

# Transform the point cloud
print(f'Lets place the object somewhere in the world.')
T = random_SE3()
in_points = torch.bmm(in_points, torch.from_numpy(T[:3, :3]).float().unsqueeze(0).to(device)) + torch.from_numpy(T[:3, 3]).float().unsqueeze(0).to(device)
points3d = in_points[0].detach().cpu().numpy()
print(f'what do we have after transforming the points?')
print(f'min {points3d.min(axis=0)}, max: {points3d.max(axis=0)}')
# visualize_point_cloud(points3d)
print('\n')



# Define camera parameters
K = np.array([[552.41140, 0, 320.0],
                [0, 552.57043, 240.0],
                [0, 0, 1]])

# Project the point cloud to the image plane
points2d = project_point_cloud(points3d, K)

# Visualize the projected point cloud
# plt.scatter(points2d[:, 0], points2d[:, 1]); plt.show()

# Define a 2D depth image
depth = np.zeros((480, 640))
depth_points = []
for i in range(points2d.shape[0]):
    x, y = int(points2d[i, 0]), int(points2d[i, 1])
    if x < 0 or x >= 640 or y < 0 or y >= 480: continue
    if depth[y, x] < points3d[i, 2]:
        depth[y, x] = points3d[i, 2]
        depth[y, x] = points3d[i, 2]
        depth_points.append(points3d[i])
depth_points = np.array(depth_points)
N_ax, dim_ax = 0, -1  # Those are the axis for num_points and points_dim

# Visualize the depth image
# plt.imshow(depth); plt.show()

print(f"out of {points3d.shape[N_ax]} points, {depth_points.shape[N_ax]} points are in the depth image")

num_samples = 1000
print(f'lets sample {num_samples}')

rng = np.random.default_rng()
subset_idxs = rng.choice(depth_points.shape[N_ax], num_samples, replace=False)
subset = depth_points[subset_idxs]
print(f'subset shape: {subset.shape}')
print('\n')


# Center the depth point cloud
calc_offset = np.mean(subset, axis=N_ax)
subset -= calc_offset

print(f'lets shift points to center')
print(f"The mean is {calc_offset} but the original offset is {T[:3, 3]}")
print(f"Now the goal is to sample transforms and try to learn how to recover original offset.")


print('\n')
print(f'Lets scale the points to fit in a unit cube [0, 1]')

# Apply calculated scale
calc_scale = np.fabs(subset.min(axis=0) - subset.max(axis=N_ax)).max()
subset /= calc_scale 

print(f'we calculated a scale of {calc_scale}')
print(f'resultant points range: {subset.min(axis=0)} - {subset.max(axis=0)}')
print(f'\n')


# Lets set up the model portion


x = torch.as_tensor(subset).to(device).unsqueeze(0)


import torch
from torch import nn

from models.encoders.pointnet import PointNetEncoder

q_dim, z_dim, in_dim = 256, 3*4, 3

project = nn.Sequential(
    nn.Linear(q_dim, 128),
    nn.BatchNorm1d(128),
    nn.GELU(),
    nn.Linear(128, 64),
    nn.BatchNorm1d(64),
    nn.GELU(),
    nn.Linear(64, z_dim),
).to(device)

encode = PointNetEncoder(
    q_dim, 
    in_dim
).to(device)

# TEMP
project.eval(), encode.eval()

q = encode(x)[0]  # Selecting the mean
z = project(q)
T_pred = z.reshape(-1, 3, 4)

print(f'we got transform \n{T_pred}')

pred_subset = T_pred[:, :, 0:3] @ x.transpose(1, 2) + T_pred[:, :, 3:4]
print(pred_subset.shape)

print(f'now we need to try and train this to predict')
