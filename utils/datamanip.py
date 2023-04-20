from scipy.spatial.transform import Rotation
import numpy as np


def project_point_cloud(points, K):
    # Project the points to the image plane
    points = (K @ points.T).T
    points = points[:, :3] / points[:, -1:]
    return points[:, :2]



def random_SE3():
    # Random rotation
    r = Rotation.random()
    R = r.as_matrix()
    # Random translation
    t = np.random.rand(3) * 2 - 1
    t[-1] = 10
    # Random SE(3) transformation
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    # Scale the transformation
    T[:3, :3] *= 1.25
    return T


import torch
from pytorch3d.transforms import quaternion_to_matrix as quat2mat
from pytorch3d.transforms import random_quaternions as rand_quat

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

def get_random_qts(B=1, catted=False):
    quat = rand_quat(B)
    t = torch.rand(B, 3)*0.25
    s = torch.rand(B, 1)*0.25 + 1.0
    if catted:
        return torch.cat([quat, t, s], dim=1)
    return quat, t, s
