import torch
from torch import nn
from pytorch3d.structures import Pointclouds

from pytorch3d.renderer import look_at_view_transform
from pytorch3d.renderer import FoVPerspectiveCameras, FoVOrthographicCameras
from pytorch3d.renderer import PointsRasterizationSettings 
from pytorch3d.renderer import PointsRasterizer
from pytorch3d.renderer import PointsRenderer
from pytorch3d.renderer import AlphaCompositor

def rands_in_range(range, count):
    rand = torch.rand(count)
    return rand * (range[1] - range[0]) + range[0]

def sample_transforms(num_vars, 
                      dist_range=[0.8, 1.5],  
                      elev_range=[0, 360],  
                      azim_range=[0, 360],
                      device='cuda'):
    Rs, Ts = look_at_view_transform(
        rands_in_range(dist_range, num_vars),
        rands_in_range(elev_range, num_vars),
        rands_in_range(azim_range, num_vars),
        device=device
    )
    return Rs, Ts

class PointRgbdRenderer(PointsRenderer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def forward(self, point_clouds, **kwargs) -> torch.Tensor:
        # TODO: Cite the orihinal implementation
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
        depths = fragments.zbuf.permute(0, 3, 1, 2)[:, :1]

        return {'images':images, 
                'depths': depths,
                # 'depths': dists2[:, :, :, :]
            }


class RendererWrapper(nn.Module):
    '''Wraps Camera, Rasteraizer, and Renderer'''
    def __init__(self,
                 image_size=64,
                 pt_radius=0.04,
                 pts_per_pxl=4,
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
        self.raster_settings = PointsRasterizationSettings(
            image_size=image_size,
            radius=pt_radius,
            points_per_pixel=pts_per_pxl
        )
        self._dist_range = dist_range
        self._elev_range = elev_range
        self._azim_range = azim_range

    def _get_renderer(self,Rs, Ts, device):
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
            compositor=AlphaCompositor()
        )
        return renderer

    def _sample_transforms(self, n):
        return sample_transforms(n, self._dist_range, self._elev_range, 
                                 self._azim_range,)

    def __call__(self, verts, feat, num_variations=None, Rs=None, Ts=None):
        '''
        This function currently is intended to render a single cloud from varied views within a range.
        Args:
            Verts, feats: [B x N x {3, F}]
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

        pts = Pointclouds(points=verts.repeat(num_variations, 1, 1), 
                          features=feat.repeat(num_variations, 1, 1))
        render = self._get_renderer(Rs, Ts, pts.device)
        return render(pts)
