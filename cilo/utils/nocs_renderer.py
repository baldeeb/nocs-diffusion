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

class NOCSRenderer(nn.Module):
    def __init__(self,
                 image_size=512,
                 pt_radius=0.03,
                 pts_per_pxl=10,
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



    def __call__(self, verts, feat, num_variations=None, Rs=None, Ts=None):
        '''
        This function currently is intended to render a single cloud from varied views within a range.
        Args:
            Verts, feats: [num points x {3, n}]
        Return:
            number of rendered images
        '''
        # assert verts.ndim == feat.ndim == 2, "Cannot yet handle batches."
        assert num_variations is not None or not (Rs is None or Ts is None), \
            "Either num_variations or bot Rs & Ts should be set."
        
        if Rs==None or Ts==None: 
            Rs, Ts = sample_transforms(
                num_variations,
                self._dist_range, 
                self._elev_range, 
                self._azim_range, 
            ) 
        else:
            assert len(Rs) == len(Ts), "Rotations and Translations should be equal in count."
            num_variations = len(Rs)

        obj_center = (verts.max(0).values - verts.min(0).values) / 2
        centered = verts - obj_center
        if feat is None: feat = pts
        # feat = torch.concatenate([verts, feat], dim=-1)
        pts = Pointclouds(points=centered.repeat(num_variations, 1, 1), 
                          features=feat.repeat(num_variations, 1, 1))

        # Initialize a camera.        
        cameras = FoVPerspectiveCameras(
            device=pts.device,
            aspect_ratio=self._aspect_ratio,
            fov=self._fov,
            R=Rs, T=Ts,
            znear=self._znear
        )

        renderer = PointsRenderer(
            rasterizer=PointsRasterizer(
                cameras=cameras,
                raster_settings=self.raster_settings
            ),
            compositor=AlphaCompositor()
        )

        images = renderer(pts)
        return images
