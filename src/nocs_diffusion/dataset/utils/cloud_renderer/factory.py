import torch 

from pytorch3d.renderer import FoVPerspectiveCameras, FoVOrthographicCameras
from pytorch3d.renderer import PointsRasterizationSettings 
from pytorch3d.renderer import PointsRasterizer
from pytorch3d.renderer import AlphaCompositor

from .core import PointCloudRenderer

class RendererFactory:

    def __init__(self, 
                 aspect_ratio=1.0,
                 cam_fov=60,
                 znear=0.1,
                 zfar=100,
                 image_size=64,
                 pt_radius=0.05,
                 pts_per_pxl=5,
                 Renderer=PointCloudRenderer,
                 Compositor=AlphaCompositor,
                ):
        self._aspect_ratio = aspect_ratio
        self._fov = cam_fov
        self._znear = znear
        self._zfar = zfar
        self._image_size = image_size
        self.raster_settings = PointsRasterizationSettings(
            image_size=image_size,
            radius=pt_radius,
            points_per_pixel=pts_per_pxl
        )
        self.Renderer = Renderer
        self.Compositor = Compositor

    def get(self, feat_size, Rs, Ts, device):
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
        renderer = self.Renderer(
            rasterizer=rasterizer,
            compositor=self.Compositor(
                background_color=torch.ones(feat_size)),
        )
        return renderer
    
    def get_projection_matrix(self, degrees=True) -> torch.Tensor:
        return FoVPerspectiveCameras().compute_projection_matrix(
                    self._znear, 
                    self._zfar,
                    self._fov,
                    self._aspect_ratio,
                    degrees=degrees)