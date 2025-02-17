import torch
from nocs_diffusion.dataset import ShapeNetDataloader
from nocs_diffusion.dataset.shapenet import ObjRenderer, ShapeNetDataset

from time import time

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from pytorch3d.vis.plotly_vis import plot_batch_individually

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

if __name__ == '__main__':
    # Example usage
    root_dir = '/home/baldeeb/Data/ShapeNetCore'
    synset_ids = ['03797390']
    NUM_VIEWS=4
    dataset = ShapeNetDataset(root_dir, synset_ids, split='train', verbose=True)

    # Visualize meshes from the dataset
    if False:
        mesh = [dataset[i][0] for i in range(24)]
        fig = plot_batch_individually([mesh], ncols=6)
        fig.show()

    renderer = ObjRenderer(256, "cuda")
    nocs_dataset = ShapeNetDataloader(
        dataset, 
        renderer, 
        batch_size=NUM_VIEWS,
        return_dict=['images', 'masks', 'face_points', 'category_ids', "transforms"],
        )

    t0 = time()    
    renders = nocs_dataset()
    print(f'rendering took: {time() - t0}')
    # Validate nocs
    bool_mask = renders['masks'].expand(-1,3,-1,-1) == 1
    nocs_pts = renders["images"][bool_mask].view(-1, 3)
    nocs_min, nocs_max = nocs_pts.min(0).values, nocs_pts.max(0).values
    print(f"Nocs images extremes: \n\tmins: {nocs_min}\n\tmaxs: {nocs_max}")

    assert torch.all(nocs_pts.min(0).values >= -1.0), "Some nocs values are smaller than 0"
    assert torch.all(nocs_pts.max(0).values <= 1.0), "Some nocs values are larger than 1"

    # Display the images
    import matplotlib.pyplot as plt

    def show_image_grid(images, title, num_cols=3):
        num_rows = int(len(images)/num_cols)
        fig, axes = plt.subplots(num_rows, num_cols)  # Adjust figsize to your preference
        fig.suptitle(title)
        for i in range(num_rows):
            for j in range(num_cols):
                axes[i, j].imshow(images[i*num_cols + j])
                axes[i, j].axis('off')
        return fig, axes

    nocs = (renders["images"] + 1) / 2.0
    fa1 = show_image_grid(nocs.permute(0,2,3,1).cpu().numpy(),              "NOCS Images" , 2)
    fa2 = show_image_grid(renders["depths"].permute(0,2,3,1).cpu().numpy(), "Depth Images", 2)
    fa3 = show_image_grid(renders["masks"].permute(0,2,3,1).cpu().numpy(),  "Mask Images" , 2)

    # plt.show()

    img2pcd_idxs = renders['perspective_2d_indices']
    c1 = torch.stack([p[:,  i[:, 0], i[:, 1]].T 
                    for p, i in zip(renders['images'], img2pcd_idxs)])

    c2 = renders['face_points']
    c2 = renders['transforms'].transform_points(c2)

    c1 = c1.cpu().numpy()
    c2 = c2.cpu().numpy()
    display_clouds(c1[0], c2[0], names=['NOCS Image', 'Face Points'])

    plt.show()
    pass


