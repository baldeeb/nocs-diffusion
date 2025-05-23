import torch

from .dataset import ShapeNetDataset
from .collate import CollateFunctor
from . import DEFAULT_SPLIT_PERCENTAGES

from ..utils import (
    MeshRenderer, 
    RandomViewPointCloudRenderer,
    PointCloudRenderer,
    mask_from_depth, 
    list_to_torch_mat,
    )
from ..synsetids import cate_to_synsetid

class ShapeNetDataloader:
    def __init__(self, 
                 dataset,
                 renderer,
                 collate_fn=None,
                 batch_size=1,
                 points_per_render=1000,
                 return_dict=['images', 'masks', 'face_points', 'category_ids'],
                 augment_face_points=lambda _:_,
                 categories_to_ids=None,
                 ):
        self.point_per_render = points_per_render
        self.return_dict = return_dict
        self.dataset = dataset
        self.renderer = renderer
        self.batch_size = batch_size
        self.augment_face_points = augment_face_points
        self.categories_to_ids = categories_to_ids
        
        if collate_fn is None:
            self.collate_fn = CollateFunctor(
                meshes=isinstance(renderer, MeshRenderer), 
                meta_keys=['scale'])

    @staticmethod
    def build(
              # Dataset
              root_dir, 
              categories,
              split='train', 
              split_percentages=DEFAULT_SPLIT_PERCENTAGES,
              verbose=False,
              preload_objs=False,

              # Renderer
              image_size=256, 
              device='cuda', 
              
              # Dataloader   
              batch_size=1,
              point_per_render=200,
              return_dict=['images', 'masks', 'face_points', 'category_ids', 'depths'],
              augment_face_points=lambda _:_,
              category_ids=None
            ):
        
        dataset = ShapeNetDataset(root_dir, 
                                  synset_names=categories, 
                                  split=split,
                                  split_percentages=split_percentages,
                                  verbose=verbose,
                                  preload=preload_objs
                                )
        renderer = MeshRenderer(image_size,
                               device)

        categories_to_ids = {cate_to_synsetid[c]:i for c,i in zip(categories, category_ids)}
        return ShapeNetDataloader(dataset, 
                                  renderer,
                                  batch_size,
                                  point_per_render,
                                  return_dict,
                                  augment_face_points=augment_face_points,
                                  categories_to_ids=categories_to_ids)

    def __len__(self):
        return len(self.dataset)
    
    def __iter__(self):
        return self

    def __call__(self, idxs=None):
        result = {}
        renders = self.__getitem__(idxs)
        if 'transforms' in self.return_dict:
            result['transforms'] = renders['transforms']
        if 'masks' in self.return_dict:
            depths = renders['depth_images']
            result['masks'] = mask_from_depth(depths, inverse=False)
            result['masks'] = result['masks']
        if 'face_points' in self.return_dict:
            fp = renders['face_points']  # List_of_B[N_i, 3]
            fp, sampled_idxs = list_to_torch_mat(fp, self.point_per_render)
            fp = self.augment_face_points(fp)
            result['face_points'] = fp
            if 'perspective_2d_indices':
                # For the sake of clarity and completeness; these values are 
                # indices, in a 2d perspective projection, directly corresponding
                # to the 3d points in the face_points cloud. This allows one to 
                # sample the perspecvive images and derive correlations between 
                # predicted pixels and 3d porendersnts rapidly.
                pts_2d_idxs = renders['face_pts_2d_idxs']
                B, N = sampled_idxs.shape[:2]
                p2di = torch.ones((B, N, *pts_2d_idxs[0].shape[1:]))
                for i, obj_2d_idxs in enumerate(pts_2d_idxs):
                    p2di[i] = obj_2d_idxs[sampled_idxs[i]]
                result['perspective_2d_indices'] = p2di.long()
        if 'images' in self.return_dict:
            result['images'] = renders['nocs_images']
            result['images'] = ( result['images'] * 2 ) - 1
        if 'category_ids' in self.return_dict:
            result['category_ids'] = renders['category_ids']
            result['category_ids'] = result['category_ids'][:, None]  # B, 1
        if 'projection_matrix' in self.return_dict:
            result['projection_matrix'] = renders['projection_matrix']
        if 'depths':
            result['depths'] = renders['depth_images']
        for v in result.values(): v
        return result

    def to(self, device):
        self.device = device
        return self

    def __getitem__(self, idxs =None, *_, **__):
        if idxs is None:
            idxs = (torch.rand(self.batch_size) - 1e-6) * self.__len__()
        
        if self.batch_size > 1:
            idxs = idxs.long()
            batch = [self.dataset[i] for i in idxs]
            objects, synset_id, file_id, meta = self.collate_fn(batch)
        else:
            objects, synset_id, file_id, meta = self.dataset[idxs]
            meta = meta['models']

        if self.categories_to_ids is not None:
            synset_id = [self.categories_to_ids[i] for i in synset_id]

        renders = self.renderer(objects, objects, scale = meta.get('scale', 1.0))
        renders.update({"category_ids": torch.tensor([int(i) for i in synset_id]).float(),
                         "file_id": file_id,
                         "meta":meta})
        return renders
