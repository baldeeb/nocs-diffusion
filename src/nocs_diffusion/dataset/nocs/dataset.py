import torch
from torch.utils.data import Dataset
from pathlib import Path

class RenderedNocsDataset(Dataset):
    def __init__(self, dataset_dir, transform=None, device='cpu'):
        """
        Args:
            dataset_dir (str or Path): Path to the folder containing .pt files.
            transform (callable, optional): Optional transform to be applied on the data.
            device (str): Device to load the data onto ('cpu' or 'cuda').
        """
        self.dataset_dir = Path(dataset_dir)
        self.files = sorted(self.dataset_dir.glob("*.pt"))  # List all .pt files
        self.transform = transform
        self.device = device

        if not self.files:
            raise ValueError(f"No .pt files found in directory: {self.dataset_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # Load the .pt file
        file_path = self.files[idx]
        data = torch.load(file_path, map_location=self.device)

        # Extract metadata from the filename
        file_name = file_path.stem  # Get the filename without extension
        synset_id, object_id, render_id = file_name.split("_")
        render_id = int(render_id)

        # Add metadata to the data dictionary
        data['synset_id'] = synset_id
        data['object_id'] = object_id
        data['render_id'] = render_id

        # Apply any transformations if provided
        if self.transform:
            data = self.transform(data)

        return data
    
    @staticmethod
    def collate_fn(batch):
        """
        Custom collate function to handle variable-sized tensors in the batch.
        """
        # Stack images, depths, and masks
        images = torch.cat([item['images'] for item in batch])
        depths = torch.cat([item['depths'] for item in batch])
        masks = torch.cat([item['masks'] for item in batch])

        # Create a dictionary to hold the batch data
        batch_data = {
            'images': images,
            'depths': depths,
            'masks': masks,
            'class_id': torch.tensor([item['class_id'] for item in batch]),
            'synset_id': [item['synset_id'] for item in batch],
            'object_id': [item['object_id'] for item in batch],
            'render_id': [item['render_id'] for item in batch]
        }

        return batch_data