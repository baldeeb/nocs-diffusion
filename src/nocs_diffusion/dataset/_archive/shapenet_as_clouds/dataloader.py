
from ..utils import list_to_torch_mat, mask_from_depth
from .shapenet_renderer import ShapeNetRenderer

import torch

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
            fp = renders['face_points']  # List(B)[N_i, 3]
            fp, sampled_idxs = list_to_torch_mat(fp, self.point_per_render)  # B, N, 3
            fp = self.augment_face_points(fp)
            result['face_points'] = fp.to(self.device)
            if 'perspective_2d_indices':
                # For the sake of clarity and completeness; these values are 
                # indices, in a 2d perspective projection, directly corresponding
                # to the 3d points in the face_points cloud. This allows one to 
                # sample the perspecvive images and derive correlations between 
                # predicted pixels and 3d porendersnts rapidly.
                pts_2d_idxs = renders['face_pts_2d_idxs']  # List[Tensor_i(N_i, 2)]
                B, N = sampled_idxs.shape[:2]
                p2di = torch.ones(B, N, *pts_2d_idxs[0].shape[1:]) # B, N, 2
                for i, obj_2d_idxs in enumerate(pts_2d_idxs):
                    p2di[i] = obj_2d_idxs[sampled_idxs[i]]
                result['perspective_2d_indices'] = p2di.long().to(self.device)  # B, N, 2
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