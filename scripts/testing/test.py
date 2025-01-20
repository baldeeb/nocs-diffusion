import torch
from nocs_diffusion.dataset import ShapeNetDataloader
from nocs_diffusion.dataset.shapenet import ObjRenderer, ShapeNetDataset


from time import time

if __name__ == '__main__':
    # Example usage
    root_dir = '/home/baldeeb/Data/ShapeNetCore'
    synset_ids = ['03797390']
    NUM_VIEWS=4
    dataset = ShapeNetDataset(root_dir, synset_ids, split='train', verbose=True)
    renderer = ObjRenderer(256, "cuda")
    nocs_dataset = ShapeNetDataloader(dataset, renderer, batch_size=NUM_VIEWS)

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

    plt.show()
    pass

