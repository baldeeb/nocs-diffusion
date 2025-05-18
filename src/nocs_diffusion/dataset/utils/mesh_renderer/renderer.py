# NOTE
# This code is incomplete and buggy.
#   1. Need to verify that assumptions on initial mesh 
#      size limits and centering are correct.
#   2. Deriving NOCS coordinates is BAD / WRONG.
#   3. Transform applications need to ve validated.

import torch

from pytorch3d.renderer import (
    FoVPerspectiveCameras, 
    RasterizationSettings, 
    MeshRendererWithFragments,
    MeshRasterizer, 
    HardGouraudShader,
    AmbientLights,
)
from pytorch3d.renderer.materials import Materials
from pytorch3d.renderer.cameras import look_at_view_transform


class MeshRenderer:
    def __init__(self, image_size=256, get_intrinsics=False, get_points=False):
        """
            image_size: int, size of the image to be rendered
            get_intrinsics: bool, whether to return the camera intrinsics
            get_points: bool, whether to return the 3D points
        """
        self.get_intrinsics = get_intrinsics
        self.get_points = get_points
        self.image_size = image_size

    def __call__(self, meshes, R=None, T=None, scale=None):
        """
            meshes: pytorch3d.structures.Meshes normalized to unit sphere????
            R: torch.Tensor (b, 3, 3)
            T: torch.Tensor (b, 3)
            scale: torch.Tensor (b, 1)
        """
        self.device = meshes.device
        num_views = len(meshes)
        if R is None or T is None:
            R, T = self.sample_camera_viewing_transforms(num_views)

        if isinstance(scale, float):
            scale = torch.tensor([scale])
            if num_views > 1:
                scale = scale.expand(num_views)

        if scale is not None:
            meshes = meshes.scale_verts(scale.to(self.device))
        
        # Define the settings for rasterization and shading
        raster_settings = RasterizationSettings(
            image_size=self.image_size,
            blur_radius=0.0,
            faces_per_pixel=1,
        )

        # Initialize an OpenGL perspective camera
        cameras = FoVPerspectiveCameras(device=self.device, 
                                        R=R, T=T)
        
        # Create a renderer
        rasterizer = MeshRasterizer(cameras=cameras, 
                                    raster_settings=raster_settings)
        lighting = AmbientLights(device=self.device)
        materials = Materials(device=self.device,
                              specular_color=[[0.0, 0.0, 0.0]],
                              shininess=0.0,
                              ambient_color=((1, 1, 1),),
                              diffuse_color=[[1.0, 1.0, 1.0]])
        shader = HardGouraudShader(device=self.device,
                                   cameras=cameras,
                                   lights=lighting,
                                   materials=materials)
        renderer = MeshRendererWithFragments(rasterizer=rasterizer,
                                             shader=shader)

        # Extract Images and Depths
        images, fragments = renderer(meshes)
        images = images.permute(0, 3, 1, 2)[:, :3] # (b, 3, h, w)
        depths = fragments.zbuf[:,:,:, :1]

        # Binary mask
        masks = (depths.permute(0,3,1,2) > 0)   # (b, 1, h, w)

        # Get world to view transforms
        world_to_view_transforms = cameras.get_world_to_view_transform()

        if self.get_points:
            camera_pts, perspective_2d_idxs = self.get_3d_points(self, depths, masks, cameras,
                                                                num_views, world_to_view_transforms)

        # Permute and mask depth
        depths = depths.permute(0,3,1,2) # (b, 1, h, w)

        # # TODO get this, remove background from facepoints and the 2d-indices here.
        # # The one-dimensional-indices of the points 
        # #  seen from this perspective, as they are 
        # #  organized in the object cloud 
        # perspective_2d_idxs = [torch.where(d) for d in depth_images]

        render = {"images":  images, 
                  "depths": depths,
                  "masks":  masks,
                  "transforms": world_to_view_transforms.inverse()}
        if self.get_points:
            render.update({ "face_points":  camera_pts,
                            "face_pts_2d_idxs": perspective_2d_idxs})

        if self.get_intrinsics:
            render.update({ "projection_matrix": cameras.get_projection_transform() })
        return render

    def get_3d_points(self, depths, masks, cameras, num_views, world_to_view_transforms):
        """
            Get 3D points from the depth images and the camera intrinsics
            depths: (b, 1, h, w)
            masks: (b, 1, h, w)
            cameras: pytorch3d.renderer.FoVPerspectiveCameras
            num_views: int
            world_to_view_transforms: pytorch3d.renderer.cameras.Transform3D
        """
        # Get 3D coordinates
        ## Get ij frame coordinates
        b, h, w = depths.shape[0:3]
        i, j = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij') 
        center = torch.tensor([h/2.0, w/2.0]).view(2, 1, 1).to(self.device)
        ji = torch.stack([j, i], dim=0).to(self.device).float()
        
        ## Get normalized coordinates (check pytorch3d docs)
        ndc = (ji / center) - 1.0
        ndc = ndc[None].repeat(num_views, 1, 1, 1).permute(0, 2, 3, 1)
        ndc = torch.concatenate([ndc, depths.clone()], dim=-1)
        
        ## Transform NDC to world
        world_coordinates = cameras.unproject_points(ndc.view(b,-1, 3))
        camera_coordinates = world_to_view_transforms.transform_points(world_coordinates)
        
        ## Reshape to images
        world_coordinates  = world_coordinates.view(b, h, w, 3)
        camera_coordinates = camera_coordinates.view(b, h, w, 3)

        ## TODO: there are redundant calculations that can be avoided. Look at processing done in dataloader.
        torch.stack([j, i], dim=0).to(self.device).float()
        camera_coordinates = torch.concatenate(
            [camera_coordinates,
            i[None, :, :, None].expand(b, h, w, 1).to(self.device),
            j[None, :, :, None].expand(b, h, w, 1).to(self.device)], 
            dim=-1)
        ## TODO: do not recalculate the mask here. 
        camera_pts = [camera_coordinates[i, 
                                        masks[i].permute(1,2,0).expand(h, w, 5)
                                        ].view(-1, 5)
                    for i in range(b)]
        perspective_2d_idxs = [c[:, -2:].long() for c in camera_pts]
        camera_pts = [c[:, :3] for c in camera_pts]

        return camera_pts, perspective_2d_idxs

    def generate_random_camera_viewing_angles(self, count=1):
        # return 1.20, 45.0, 225.0
        dist = torch.rand(count) * 0.5 + 1.0   # Distance in meters
        elev = torch.rand(count) * 90        # Elevation in degrees
        azim = torch.rand(count) * 360       # Azimuth in degrees
        return dist, elev, azim


    def sample_camera_viewing_transforms(self, count):

        # Generate random camera views
        d, e, a = self.generate_random_camera_viewing_angles(count)
        R, T = look_at_view_transform(dist=d, elev=e, azim=a,
                                      device=self.device)
        return R, T