'''
Question: can we learn Rotation, Translation and Scale of an observed object
Given we know some description of that object?
'''



import torch
from habitat_datatools.utils.dataset import HabitatDataset
import numpy as np

from models.encoders.pointnet import PointNetEncoder
from models.diffusion import DiffusionPoint, VarianceSchedule

# point_enc = PointNetEncoder(12)
# def get_latent_encoder():
    
#     # Checkpoint
#     ckpt = torch.load(args.ckpt)
#     seed_all(ckpt['args'].seed)

#     # Model
#     logger.info('Loading model...')
#     model = AutoEncoder(ckpt['args']).to(args.device)
#     model.load_state_dict(ckpt['state_dict'])
#     diffusion = DiffusionPoint(
#                 net = PointwiseNet(point_dim=3, context_dim=args.latent_dim, residual=args.residual),
#                 var_sched = VarianceSchedule(
#                     num_steps=args.num_steps,
#                     beta_1=args.beta_1,
#                     beta_T=args.beta_T,
#                     mode=args.sched_mode
#                 )
#             )

if __name__ == '__main__':
    DATAPATH='/home/baldeeb/Code/nocs-diffusion/data/habitat_data/00847-bCPU9suPUw9/metadata.json'
    dataset = HabitatDataset(DATAPATH)

    object_id = 'b2d6c064af0c735f1783a44a88d6274'
    
    rgb, d, s, coord, e = dataset[0]

    # Get masked depth
    d_masked = d[s == 1]

    # project to 3D
    K = dataset.intrinsic()
    K_inv = np.linalg.inv(K)
    x, y = np.meshgrid(np.arange(d.shape[1]), np.arange(d.shape[0]))
    x = x[s == 1]
    y = y[s == 1]
    z = d_masked
    xy1 = np.stack([x, y, np.ones_like(x)], axis=1)
    xy1 = xy1 @ K_inv[:3, :3]
    xyz = xy1 * z[:, None]

    # Visualize the 3d points
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2])
    plt.show()

    # Center and normalize points
    mu, sigma = xyz.mean(axis=0), xyz.std(axis=0)
    xyz -= mu
    xyz /= sigma

    # make torch tensor [B, C, N]
    points = torch.from_numpy(xyz).float().unsqueeze(0).transpose(1, 2)
