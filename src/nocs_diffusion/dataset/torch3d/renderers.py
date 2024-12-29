

from pytorch3d.renderer import PointsRenderer
import torch

class PointRgbdRenderer(PointsRenderer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, point_clouds, **kwargs) -> torch.Tensor:
        # TODO: Cite the original implementation
        fragments = self.rasterizer(point_clouds, **kwargs)

        # Construct weights based on the distance of a point to the true point.
        # However, this could be done differently: e.g. predicted as opposed
        # to a function of the weights.
        r = self.rasterizer.raster_settings.radius

        dists2 = fragments.dists.permute(0, 3, 1, 2)

        # The one-dimensional-indices of the points 
        #  seen from this perspective, as they are 
        #  organized in the object cloud 
        indices = fragments.idx.long().permute(0, 3, 1, 2) 
        
        weights = 1 - dists2 / (r * r)
        images = self.compositor(
            indices,
            weights,
            point_clouds.features_packed().permute(1, 0),
            **kwargs,
        )

        # permute so data channels are the end
        images = images.permute(0, 2, 3, 1)
        depths = fragments.zbuf[:,:,:, :1]
        obj_clouds = point_clouds.points_packed() 

        perspective_2d_idxs = [torch.argwhere(batch_idxs  != -1) 
                               for batch_idxs in indices[:, 0]]

        # Sample the points that are visible from the used perspective
        face_pts = [obj_clouds[indices[i, 0, prspctv_i[:, 0], prspctv_i[:, 1]]] 
                    for i, prspctv_i in enumerate(perspective_2d_idxs)]
        
        return {'images':images, 
                'depths': depths, 
                'face_points':face_pts,
                'face_pts_2d_idxs': perspective_2d_idxs}
