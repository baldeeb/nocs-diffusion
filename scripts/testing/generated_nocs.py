import torch
from pytorch3d import transforms
from nocs_diffusion.utils.alignment import simple_align

import hydra
from omegaconf import DictConfig

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def display_clouds(c1, c2, names=['PointCloud 1', 'PointCloud 2']):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the first point cloud in red
    ax.scatter(c1[:, 0], c1[:, 1], c1[:, 2], alpha=0.2, c='r', marker='o', label=names[0])

    # Plot the second point cloud in blue
    ax.scatter(c2[:, 0], c2[:, 1], c2[:, 2], alpha=0.2, c='b', marker='^', label=names[1])

    # draw vectors between cloud points of the same index
    for i in range(c1.shape[0]):
        if torch.rand(1).item() > 0.1: continue
        ax.plot([c1[i, 0], c2[i, 0]], [c1[i, 1], c2[i, 1]], [c1[i, 2], c2[i, 2]], alpha=0.2)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.legend()


@hydra.main(version_base=None, config_path='../../config', 
            # config_name='train_nocs_diffuser'
            config_name='train_nocs_diffuser_new_dataloader'
            )
def run(cfg: DictConfig) -> None:
    dataloader = hydra.utils.instantiate(cfg.dataloader).to(cfg.device)
    d = dataloader()

    img2pcd_idxs = d['perspective_2d_indices']
    pts = torch.stack([p[:,  i[:, 0], i[:, 1]].T 
                    for p, i in zip(d['images'], img2pcd_idxs)])
    Rts, mean_dists, dists, scale = simple_align(d['face_points'], pts)

     
    if Rts.shape[-2] == 3: 
        last_row = torch.ones(*Rts.shape[:-2], 1, 4) * torch.tensor([0, 0, 0, 1])
        Rts = torch.concatenate((Rts, last_row.to(Rts.device)), dim=-2)
        
    I = torch.eye(4).to(Rts.device)
    gt_Rt = d['transforms'].get_matrix().to(Rts.device)
    residual_Rts = (I - ( Rts.inverse() @ gt_Rt )).norm()
    print(f'residual transform: {residual_Rts}')
    print(f'mean dist: {mean_dists}')

    ## Transform face points using the estimated transform and visualize it with the NOCS image as 3d clouds
    img2pcd_idxs = d['perspective_2d_indices']
    c1 = torch.stack([p[:,  i[:, 0], i[:, 1]].T 
                    for p, i in zip(d['images'], img2pcd_idxs)])

    c2 = d['face_points']
    pred_transform = transforms.Transform3d(matrix=Rts.transpose(-1, -2))
    c2 = pred_transform.transform_points(c2)
    c2 *= scale[:, None]

    c1 = c1.cpu().numpy()
    c2 = c2.cpu().numpy()
    display_clouds(c1[0], c2[0], names=['NOCS Image', 'Face Points'])
    plt.show()

if __name__ == '__main__':
    run()