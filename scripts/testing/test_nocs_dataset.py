import time
from tqdm import tqdm
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

from nocs_diffusion.dataset import RenderedNocsDataset

# Path to the folder containing .pt files
dataset_dir = './data/nocs_renders'

# Create the dataset
dataset = RenderedNocsDataset(dataset_dir, device='cuda')

# Create a DataLoader
dataloader = DataLoader(dataset, collate_fn=dataset.collate_fn, batch_size=8, shuffle=True)

# Iterate through the DataLoader
time_acc = []
for batch in tqdm(dataloader):
    start_time = time.time()
    assert(batch['images'].shape[-2:] == batch['depths'].shape[-2:] == batch['masks'].shape[-2:])  
    assert(batch['images'].shape[0] == batch['depths'].shape[0] == batch['masks'].shape[0])

    time_acc.append(time.time() - start_time)

print(f"Average time per batch: {sum(time_acc)/len(time_acc):.6f} seconds")

# load a batch and display the images
def show_batch(batch):
    images = batch['images']
    depths = batch['depths']
    masks = batch['masks']

    fig, axs = plt.subplots(3, 8, figsize=(20, 6))
    for i in range(8):
        axs[0, i].imshow(images[i].permute(1, 2, 0).cpu().numpy())
        axs[0, i].set_title('Image')
        axs[0, i].axis('off')

        axs[1, i].imshow(depths[i, 0].cpu().numpy(), cmap='gray')
        axs[1, i].set_title('Depth')
        axs[1, i].axis('off')

        axs[2, i].imshow(masks[i, 0].cpu().numpy(), cmap='gray')
        axs[2, i].set_title('Mask')
        axs[2, i].axis('off')

    plt.tight_layout()
    plt.show()
# Show a batch of images
for batch in dataloader:
    show_batch(batch)
    break
