import torch
from torch import nn
from pytorch3d.structures import Pointclouds

from pytorch3d.renderer import look_at_view_transform
from pytorch3d.renderer import FoVPerspectiveCameras, FoVOrthographicCameras
from pytorch3d.renderer import PointsRasterizationSettings 
from pytorch3d.renderer import PointsRasterizer
from pytorch3d.renderer import PointsRenderer
from pytorch3d.renderer import AlphaCompositor

from pytorch3d.ops import estimate_pointcloud_normals

from .renderer_tools import sample_transforms, mask_from_depth

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
        indices = fragments.idx.long().permute(0, 3, 1, 2)
        weights = 1 - dists2 / (r * r)
        images = self.compositor(
            indices,
            weights,
            point_clouds.features_packed().permute(1, 0),
            **kwargs,
        )

        # permute so image comes at the end
        images = images.permute(0, 2, 3, 1)
        depths = fragments.zbuf[:,:,:, :1]

        # package points visible in render
        face_pts = []
        points_packed = point_clouds.points_packed()
        B = indices.shape[0]
        for i in range(B):
            face_idxs = indices[i].contiguous().view(-1)
            face_pts.append(points_packed[face_idxs])

        return {'images':images, 
                'depths': depths, 
                'face_points':face_pts}


class Torch3DRendererWrapper(nn.Module):
    '''Wraps Camera, Rasteraizer, and Renderer'''
    def __init__(self,
                 image_size=64,
                 pt_radius=0.05,
                 pts_per_pxl=5,
                 aspect_ratio=torch.tensor(1.0),
                 cam_fov=torch.tensor(60),
                 znear=0.01,
                 dist_range=[0.8, 1.5], 
                 elev_range=[0, 360], 
                 azim_range=[0, 360]
                ):
        self._aspect_ratio = aspect_ratio
        self._fov = cam_fov
        self._znear = znear
        self._image_size = image_size
        self.raster_settings = PointsRasterizationSettings(
            image_size=image_size,
            radius=pt_radius,
            points_per_pixel=pts_per_pxl
        )
        self._dist_range = dist_range
        self._elev_range = elev_range
        self._azim_range = azim_range

    def _get_renderer(self, feat_size, Rs, Ts, device):
        cameras = FoVPerspectiveCameras(
            device=device, 
            aspect_ratio=self._aspect_ratio,
            fov=self._fov,
            R=Rs, T=Ts,
            znear=self._znear
        )
        rasterizer = PointsRasterizer(
            cameras=cameras,
            raster_settings=self.raster_settings
        )
        renderer = PointRgbdRenderer(
            rasterizer=rasterizer,
            compositor=AlphaCompositor(background_color=torch.ones(feat_size)),
        )
        return renderer

    def _sample_transforms(self, n):
        return sample_transforms(n, self._dist_range, self._elev_range, 
                                 self._azim_range,)

    def __call__(self, verts, feats, normals=None, num_variations=None, Rs=None, Ts=None):
        '''
        This function currently is intended to render a single cloud from varied views within a range.
        Args:
            Verts, feats: [{None , B} x N x {3, F}]
            normals
        Return:
            number of rendered images & depth images
        '''
        if Rs==None or Ts==None: 
            if num_variations is None: num_variations = 1
            Rs, Ts = self._sample_transforms(num_variations)
        elif Rs is not None and Ts is not None :
            assert len(Rs) == len(Ts), "Rotations and Translations should be equal in count."
            num_variations = len(Rs)
        else: assert False, "Either num_variations or bot Rs & Ts should be set."

        if verts.ndim == 2:  # if a single object is givens
            verts = verts.expand(num_variations, -1, -1)
            feats = feats.expand(num_variations, -1, -1)
            if normals: 
                normals = normals.expand(num_variations, -1, -1)
        
        if normals is not None:
            feats = torch.concatenate([feats, normals], dim=-1)

        pts = Pointclouds(points=verts, features=feats,)
        feature_dim = feats.shape[-1]
        render = self._get_renderer(feature_dim, Rs, Ts, pts.device)
        result = render(pts)
        
        if normals is not None:
            result['normals'] = result['images'][:, :, :, -3:]
            result['images'] = result['images'][:, :, :, :-3]
        
        return result
