import torch
from torch import nn
from pytorch3d.structures import Pointclouds

from pytorch3d.renderer import look_at_view_transform
from pytorch3d.renderer import FoVPerspectiveCameras, FoVOrthographicCameras
from pytorch3d.renderer import PointsRasterizationSettings 
from pytorch3d.renderer import PointsRasterizer
from pytorch3d.renderer import PointsRenderer
from pytorch3d.renderer import AlphaCompositor
from pytorch3d.renderer import get_world_to_view_transform

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
        indices = fragments.idx.long().permute(0, 3, 1, 2) # The one-dimensional-indices of the points 
                                                           #  seen from this perspective, as they are 
                                                           #  organized in the object cloud 
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

        perspective_2d_idxs = [torch.argwhere(batch_idxs  != -1) for batch_idxs in indices[:, 0]]

        # Sample the points that are visible from the used perspective
        face_pts = [obj_clouds[indices[i, 0, prspctv_i[:, 0], prspctv_i[:, 1]]] for i, prspctv_i in enumerate(perspective_2d_idxs)]
        
        return {'images':images, 
                'depths': depths, 
                'face_points':face_pts,
                'face_pts_2d_idxs': perspective_2d_idxs}


class Torch3DRendererWrapper(nn.Module):
    '''Wraps Camera, Rasteraizer, and Renderer'''
    def __init__(self,
                 image_size=64,
                 pt_radius=0.05,
                 pts_per_pxl=5,
                 aspect_ratio=torch.tensor(1.0),
                 cam_fov=torch.tensor(60),
                 znear=0.01,
                 zfar=100.0,
                 dist_range=[0.8, 1.5], 
                 elev_range=[0, 360], 
                 azim_range=[0, 360],
                 points_in_perspective=True,
                 **kwargs
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
        self._dist_range = dist_range
        self._elev_range = elev_range
        self._azim_range = azim_range

        self.points_in_perspective = points_in_perspective

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

    def get_projection_matrix(self) -> torch.Tensor:
        self._K = FoVPerspectiveCameras().compute_projection_matrix(
            self._znear, self._zfar, self._fov, self._aspect_ratio, degrees=True)
        return self._K

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
        
        if self.points_in_perspective:
            ''' Not very confident in this but here is the rationale:
            The defined Rs and Ts are applied to the camera relative to 
            some frame. Those are akin to applying the inverse of those
            transforms to the cloud. To position the points relative to 
            the camera we could simply apply the same transform to the
            cloud, undoing the effect of camera rotation but positioning
            the cloud points in a location similar to what is expected 
            if we only have reprojected dense depth images.
            All this is not really validated but the intention is to 
            test its effectiveness through experiemtns.'''
            def get_transformed_pts():
                for R, t, pts in zip(Rs, Ts, result['face_points']):
                    yield get_world_to_view_transform(R[None], t[None]).transform_points(pts[None])[0]
            result['face_points'] = [p for p in get_transformed_pts()]
            # Rt = get_world_to_view_transform(Rs, Ts)
            # result['face_points'] = [Rti.transform_points(p) for Rti, p in zip(Rt, result['face_points']) ]
        result['Rts'] = get_world_to_view_transform(Rs, Ts)

        return result
