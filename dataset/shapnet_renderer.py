from .shapenet import ShapeNetCore
from .renderer import RendererWrapper, sample_transforms, mask_from_depth
from .nocs_tools import nocs_extractor

import torch

class ShapeNetRenderLoader:
    def __init__(self, 
                 dataset:ShapeNetCore, 
                 renderer:RendererWrapper, 
                 num_objects:int, 
                 batch_size:int,
                 feature_extractor=nocs_extractor):
        self.dataset = dataset
        self.renderer = renderer
        self.num_objects = num_objects
        self.batch_size = batch_size

        obj_ids = torch.randperm(len(dataset))[:num_objects]
        self.point_clouds = self.dataset[obj_ids]
        self.point_features = feature_extractor(self.point_clouds)


    def _sample_batch_of_clouds(self):
        idxs = torch.multinomial(torch.range(0, self.num_objects), 
                                 self.batch_size, 
                                 replacement=True)
        clouds = self.point_clouds[idxs]
        features = self.point_features[idxs]
        return clouds, features
        

    def __call__(self):
        Rs, Ts = sample_transforms(self.batch_size, elev_range=[0, 70])
        clouds, features = self._sample_batch_of_clouds()
        renders = self.renderer(clouds, features, Rs=Rs, Ts=Ts)
        return renders
        

    @staticmethod
    def build(path, 
              categories, 
              split, 
              scale_mode, 
              image_size, 
              batch_size, 
              num_objects, 
              feature_extractor=nocs_extractor):
        dataset = ShapeNetCore(path=path,
                                cates=categories,
                                split=split,
                                scale_mode=scale_mode)
        render = RendererWrapper(image_size=image_size)
        return ShapeNetRenderLoader(dataset, render, batch_size, num_objects, feature_extractor=feature_extractor)