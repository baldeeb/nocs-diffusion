import torch
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm

from nocs_diffusion.dataset.shapenet import NocsColoredShapeNetDataset


if __name__ == '__main__':
    # Example usage
    root_dir = './data/nocsObjModels'
    synset_ids = ['03797390']
    renders_per_object=50
    image_size = 32

    output_dir = Path('./data/nocs_renders').absolute()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset = NocsColoredShapeNetDataset(
                root_dir, synset_ids, split='train',
                image_size = image_size,)
    
    for data_idx in tqdm(range(len(dataset)), desc='Objects', position=0):
        for render_idx in tqdm(range(renders_per_object), desc='Renders', position=1, leave=False):
            results = dataset[data_idx]
            synset_id = results['synset_id']
            object_id = results['file_id']
            torch.save(results, output_dir / f'{synset_id}_{object_id}_{render_idx}.pt')