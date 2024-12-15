from .shapenet.shapenet import ShapeNetCore, cate_to_synsetid
from .renderer import Torch3DRendererWrapper, sample_transforms, mask_from_depth
from .renderer_tools import sample_from_clouds, sample_dim_zero, list_to_torch_mat
from .nocs_tools import nocs_extractor

import torch

class ShapeNetRenderer:
    def __init__(self, 
                 dataset:ShapeNetCore, 
                 renderer:Torch3DRendererWrapper, 
                 num_objects:int, 
                 batch_size:int,
                 shuffle:bool=True,
                 feature_extractor=nocs_extractor,
                 cloud_post_process=lambda _:_,
                 device='cpu',
                 categories_to_ids=None,
                 **_):
        self.dataset = dataset
        self.renderer = renderer
        self.num_objects = num_objects
        self.batch_size = batch_size
        self.feature_extractor = feature_extractor
        if categories_to_ids is None:
            self.categories_to_ids = {c:int(i) for c, i in cate_to_synsetid.items()}
        else:
            self.categories_to_ids = categories_to_ids
        self._preload_clouds(shuffle)
        self.to(device)
        self.cloud_post_process = cloud_post_process

    def _preload_clouds(self, shuffle):
        if self.num_objects is None: self.num_objects = len(self.dataset)
        if shuffle: obj_ids = torch.randperm(len(self.dataset), dtype=torch.long)
        else: obj_ids = torch.arange(0, len(self.dataset), dtype=torch.long)
        pcd_data = self.dataset.pointclouds
        select_ids = obj_ids[:self.num_objects]
        self.clouds = torch.stack([pcd_data[i]['pointcloud'] for i in select_ids])
        categories = [pcd_data[i]['cate'] for i in select_ids]
        self.cate_ids = torch.tensor([self.categories_to_ids[c] for c in categories])
        self.features = self.feature_extractor(self.clouds)

    def to(self, device):
        self.clouds = self.clouds.to(device)
        self.features = self.features.to(device)
        self.cate_ids = self.cate_ids.to(device)
        return self
    
    def _sample_batch_of_clouds(self):
        idxs = torch.multinomial(torch.ones(self.num_objects), 
                                 self.batch_size, 
                                 replacement=True)
        clouds = self.cloud_post_process(self.clouds[idxs])
        features = self.features[idxs]
        cates = torch.stack([self.cate_ids[i] for i in idxs]).float()[:, None]
        return clouds, features, cates

    def __call__(self):
        # TODO: make Transform sampler a functor and pass it in after configuring it.
        Rs, Ts = sample_transforms(self.batch_size, 
                                   elev_range=[0, 70], 
                                   device=self.clouds.device)
        clouds, features, cates = self._sample_batch_of_clouds()
        renders = self.renderer(clouds, features, Rs=Rs, Ts=Ts)
        renders['category_ids'] = cates
        return renders

    @staticmethod
    def build(path, 
              categories, 
              category_ids,
              split, 
              scale_mode, 
              image_size, 
              **kwargs):
        dataset = ShapeNetCore(path=path,
                                cates=categories,
                                split=split,
                                scale_mode=scale_mode)
        renderer = Torch3DRendererWrapper(image_size=image_size, **kwargs)
        
        cates_to_ids = {c:i for c, i in zip(categories, category_ids)}
        return ShapeNetRenderer(dataset=dataset, 
                                renderer=renderer, 
                                categories_to_ids= cates_to_ids,
                                **kwargs)


class ShapeNetDataloader:
    def __init__(self, renderer:ShapeNetRenderer, return_dict, points_per_render, augment_face_points=lambda _:_):
        self.shapenet_renderer = renderer
        self.point_per_render = points_per_render
        self.augment_face_points = augment_face_points
        self.return_dict = return_dict  # TODO: rename. this is a list not a dict. 
        self.device='cuda'

    def to(self, device):
        self.shapenet_renderer.to(device)
        self.device = device
        return self

    def __call__(self):
        result = {}
        renders = self.shapenet_renderer()
        if 'transforms' in self.return_dict:
            result['transforms'] = renders['Rts']
        if 'masks' in self.return_dict:
            depths = renders['depths'].permute(0, 3, 1, 2)
            result['masks'] = mask_from_depth(depths, inverse=False)
            result['masks'].to(self.device)
        if 'face_points' in self.return_dict:
            fp = renders['face_points']
            fp, sampled_idxs = list_to_torch_mat(fp, self.point_per_render)
            fp = self.augment_face_points(fp)
            result['face_points'] = fp.to(self.device)
            if 'perspective_2d_indices':
                # For the sake of clarity and completeness; these values are 
                # indices, in a 2d perspective projection, directly corresponding
                # to the 3d points in the face_points cloud. This allows one to 
                # sample the perspecvive images and derive correlations between 
                # predicted pixels and 3d porendersnts rapidly.
                pts_2d_idxs = renders['face_pts_2d_idxs']
                B, N = sampled_idxs.shape[:2]
                p2di = torch.ones(B, N, *pts_2d_idxs[0].shape[1:])
                for i, obj_2d_idxs in enumerate(pts_2d_idxs):
                    p2di[i] = obj_2d_idxs[sampled_idxs[i]]
                result['perspective_2d_indices'] = p2di.long().to(self.device)
        if 'images' in self.return_dict:
            result['images'] = renders['images'].permute(0, 3, 1, 2)
            result['images'] = ( result['images'] * 2 ) - 1
            result['images'].to(self.device)
        if 'category_ids' in self.return_dict:
            result['category_ids'] = renders['category_ids'].to(self.device)
        if 'projection_matrix' in self.return_dict:
            result['projection_matrix'] = self.shapenet_renderer.renderer.get_projection_matrix().to(self.device)
        
        for v in result.values(): v.to(self.device)
        return result