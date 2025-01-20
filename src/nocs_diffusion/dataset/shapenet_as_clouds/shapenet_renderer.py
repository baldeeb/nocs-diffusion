from .shapenet import ShapeNetCore, cate_to_synsetid
from .torch3d.renderer_wrapper import Torch3DRendererWrapper, sample_transforms
from ..utils import nocs_extractor

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

