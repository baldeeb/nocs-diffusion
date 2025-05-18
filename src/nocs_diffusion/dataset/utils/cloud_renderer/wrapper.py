import torch

from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import AlphaCompositor
from pytorch3d.renderer import get_world_to_view_transform

from .core import PointCloudRenderer
from .factory import RendererFactory
from ...utils import sample_transforms

class RandomViewPointCloudRenderer(torch.nn.Module):
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
                 **_
                ):
        self.renderer_constructor = RendererFactory(
            aspect_ratio=aspect_ratio,
            cam_fov=cam_fov,
            znear=znear,
            zfar=zfar,
            image_size=image_size,
            pt_radius=pt_radius,
            pts_per_pxl=pts_per_pxl,
            Renderer=PointCloudRenderer,
            Compositor=AlphaCompositor
        )
        self._dist_range = dist_range
        self._elev_range = elev_range
        self._azim_range = azim_range

        self.points_in_perspective = points_in_perspective

    def _sample_transforms(self, n):
        return sample_transforms(n, self._dist_range, self._elev_range, 
                                 self._azim_range,)

    def __call__(self, verts, feats,
                 normals=None,
                 num_variations=None,
                 Rs=None, Ts=None,
                 scale=None):
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
        
        if isinstance(scale, float):
            scale = torch.tensor([scale])

        if scale is not None:
            verts = verts * scale.to(verts.device)[:, None, None]

        if verts.ndim == 2:  # if a single object is givens
            verts = verts.expand(num_variations, -1, -1)
            feats = feats.expand(num_variations, -1, -1)
            if normals: 
                normals = normals.expand(num_variations, -1, -1)
        
        if normals is not None:
            feats = torch.concatenate([feats, normals], dim=-1)

        pts = Pointclouds(points=verts, features=feats,)
        feature_dim = feats.shape[-1]
        render = self.renderer_constructor.get(feature_dim, Rs, Ts, pts.device)
        self._K = self.renderer_constructor.get_projection_matrix()
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
                    tf = get_world_to_view_transform(R[None], t[None])
                    tf = tf.to(pts.device)
                    yield tf.transform_points(pts[None])[0]
            result['face_points'] = [p for p in get_transformed_pts()]
            # Rt = get_world_to_view_transform(Rs, Ts)
            # result['face_points'] = [Rti.transform_points(p) for Rti, p in zip(Rt, result['face_points']) ]
        result['transforms'] = get_world_to_view_transform(Rs, Ts)

        return result
