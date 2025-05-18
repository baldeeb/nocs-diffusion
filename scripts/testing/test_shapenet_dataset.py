import torch
from nocs_diffusion.dataset.shapenet import ShapeNetDataset, ShapeNetDataloader, NocsColoredShapeNetDataset
from nocs_diffusion.dataset.utils import MeshRenderer, RandomViewPointCloudRenderer

from time import time

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

import matplotlib.pyplot as plt

from pytorch3d.vis.plotly_vis import plot_batch_individually
from pytorch3d.renderer.cameras import look_at_view_transform
from torch.utils.data import DataLoader

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


def visualize(images, masks, depths, class_id):
    # Validate nocs
    bool_mask = masks.expand(-1,3,-1,-1) == 1
    nocs_pts = images[bool_mask]
    nocs_min, nocs_max = nocs_pts.min(0).values, nocs_pts.max(0).values
    print(f"Nocs images extremes: \n\tmins: {nocs_min}\n\tmaxs: {nocs_max}")

    assert torch.all(nocs_pts.min(0).values >= -1.0-1e-4), "Some nocs values are smaller than 0"
    assert torch.all(nocs_pts.max(0).values <= +1.0+1e-4), "Some nocs values are larger than 1"

    # Visualize meshes from the dataset
    if False:
        mesh = [dataset[i][0] for i in range(24)]
        fig = plot_batch_individually([mesh], ncols=6)
        fig.show()

    def show_image_grid(images, title, num_cols=3):
        num_rows = int(len(images)/num_cols)
        fig, axes = plt.subplots(num_rows, num_cols)  # Adjust figsize to your preference
        fig.suptitle(title)
        for i in range(num_rows):
            for j in range(num_cols):
                axes[i, j].imshow(images[i*num_cols + j])
                axes[i, j].axis('off')
        return fig, axes

    fa1 = show_image_grid(images.permute(0,2,3,1).cpu().numpy(), "NOCS Images" , 2)
    fa2 = show_image_grid(depths.permute(0,2,3,1).cpu().numpy(), "Depth Images", 2)
    fa3 = show_image_grid(masks.permute(0,2,3,1).cpu().numpy(),  "Mask Images" , 2)

    plt.show()

def test_dataloader():
    dataset = NocsColoredShapeNetDataset(
                root_dir, synset_ids, split='train',
                image_size = IMAGE_SIZE,)
    
    dataloader = DataLoader(
        dataset,
        batch_size=NUM_VIEWS,
        shuffle=True,
        num_workers=0,
        collate_fn=NocsColoredShapeNetDataset.collate_fn
    )

    data_iter = iter(dataloader)
    

    t0 = time()    
    result = data_iter.__next__()
    print(f'rendering took: {time() - t0}')
        
    visualize(
        result['images'], 
        result['masks'], 
        result['depths'], 
        result['class_ids']
    )

def test_pieces():
    t0 = time()    

    dataset = ShapeNetDataset(
        root_dir, synset_ids, split='train',
        verbose=True, device=device)
    
    renderer = MeshRenderer(IMAGE_SIZE)

    obj, synset_id, file_id, meta_data = dataset[DATA_IDX]
    meshes = obj.get_pytorch3d_mesh(broadcast=NUM_VIEWS)
    
    # Sample camera viewing transforms
    dist = [1.0 for _ in range(NUM_VIEWS)]
    elev = [45 for _ in range(NUM_VIEWS)]
    azim = [a for a in np.linspace(0, 360, NUM_VIEWS)]
    Rs, Ts = look_at_view_transform(dist=dist, elev=elev, azim=azim,
                                    device=device)
    renders = renderer(meshes, R=Rs, T=Ts)

    print(f'rendering took: {time() - t0}')
    
    visualize(
        renders['images'], 
        renders['masks'], 
        renders['depths'], 
        synset_id
    )

if __name__ == '__main__':
    # Example usage
    # root_dir = '/home/baldeeb/Data/ShapeNetCore'
    root_dir = './data/nocsObjModels'
    synset_ids = ['03797390']
    NUM_VIEWS=12
    IMAGE_SIZE = 64
    DATA_IDX = 1
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # test_pieces()
    test_dataloader()
  

