from .dataset import ShapeNetDataset
from ..utils import MeshRenderer

import torch 
from torch.utils.data import Dataset

class NocsColoredShapeNetDataset(ShapeNetDataset, Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.image_size = kwargs.get('image_size', 256)
        self.device     = kwargs.get('device', 'cpu')

        self.synset_id_to_idx = {s: i for i, s in enumerate(self.synset_ids)}

        self.renderer = MeshRenderer(self.image_size)


    def __getitem__(self, idx):
        obj, synset_id, file_id, _ =  super().__getitem__(idx)
        meshses = obj.get_pytorch3d_mesh()
        renders = self.renderer(meshses)
        return self._pakage_renders(renders, synset_id, file_id)


    def get_n_renders(self, idx, num_renders):
        obj, synset_id, file_id, _ =  super().__getitem__(idx)
        meshses = obj.get_pytorch3d_mesh(broadcast=num_renders)
        renders = self.renderer(meshses)
        return self._pakage_renders(renders, [synset_id]*num_renders, [file_id]*num_renders)


    def _pakage_renders(self, renders, synset_id, file_id):
        return {'images':   renders['images'],
                'depths':   renders['depths'],
                'masks':    renders['masks'],
                'synset_id': synset_id,
                'file_id':  file_id,
                'class_id': self.synset_id_to_idx[synset_id],}


    @staticmethod
    def collate_fn(batch):
        images = torch.cat([b['images'] for b in batch])
        masks  = torch.cat([b['masks'] for b in batch])
        depths = torch.cat([b['depths'] for b in batch])
        class_ids = torch.tensor([b['class_id'] for b in batch])[:, None]

        return {
            'images':   images,
            'masks':    masks,
            'depths':   depths,
            'class_ids': class_ids
        }