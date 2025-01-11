import os
from pathlib import Path
import zipfile
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.structures.meshes import Meshes
from pytorch3d.structures import join_meshes_as_batch
import json

import torch
import numpy as np
from pytorch3d.renderer import (
    FoVPerspectiveCameras, RasterizationSettings, 
    MeshRenderer, MeshRendererWithFragments,
    MeshRasterizer, SoftPhongShader, TexturesVertex,
    PerspectiveCameras
)
from pytorch3d.renderer.cameras import look_at_view_transform
from nocs_diffusion.dataset.shapenet.synsetids import synsetid_to_cate

DEFAULT_SPLIT_PERCENTAGES = {
    'train': 0.7,
    'val': 0.15,
    'test': 0.15
}

class ShapeNetDataset(Dataset):
    def __init__(self,
                 root_dir, # Path to the ShapeNet dataset
                 synset_ids, # List of synset IDs to load 
                 split='train', # {'train', 'val', 'test'} 
                 split_percentages=DEFAULT_SPLIT_PERCENTAGES, # {Train: %, Val: %, Test: %}
                 device='cpu', # Allows user to laod all meshes to a device
                 verbose=False # Print loading information
                ):
        self.device = device
        self.root_dir = root_dir
        self.synset_ids = synset_ids
        self.split = split
        self.split_percentages = split_percentages

        if verbose:
            print(f"Loading ShapeNet dataset with synset IDs: " + 
                  f"{[synsetid_to_cate[id] for id in self.synset_ids]}")
        
        self.data, self.meta = self._load_data()
        self.split_ids = self._load_split_meta()

        if verbose:
            print(f"Loaded {len(self.split_ids[self.split])} samples for split {self.split}")

    def _load_split_meta(self):
        '''Loads the split file for the given synset IDs, 
        if the user is requesting the same IDs previously seen. 
        Otherwise, creates a pt file with the split.'''
        requested_synsets = '_'.join(self.synset_ids)
        split_file = os.path.join(self.root_dir, f"meta_dataloader_{requested_synsets}.pt")
        if os.path.exists(split_file):
            split_meta = torch.load(split_file)
        else:
            split_meta = {'train':[], 'val':[], 'test':[]}
            
            for synset_id in self.synset_ids:
                n_samples = len(self.data[synset_id])
                idxs = torch.randperm(n_samples).numpy()
                
                train_count = int(n_samples * self.split_percentages['train'])
                val_count = int(n_samples * self.split_percentages['val'])
                
                synset_file_ids = list(self.data[synset_id].keys())
                split_meta['train'] = [(synset_id, synset_file_ids[i]) 
                                       for i in idxs[:train_count]]
                split_meta['val']   = [(synset_id, synset_file_ids[i]) 
                                       for i in idxs[train_count: train_count + val_count]]
                split_meta['test']  = [(synset_id, synset_file_ids[i])
                                       for i in idxs[train_count + val_count:]]
                torch.save(split_meta, split_file)
        
        return split_meta
    
    def _load_data(self):
        data = {}
        meta = {}
        for synset_id in self.synset_ids:
            data[synset_id] = {}
            meta[synset_id] = {}
            folder_path = os.path.join(self.root_dir, f"{synset_id}")
            if not os.path.isdir(folder_path):
                zip_path = f"{folder_path}.zip"
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(self.root_dir)
            
            obj_files = [f"{d}/{f}" for d, _, files in os.walk(folder_path) 
                                    for f in files 
                                    if f.endswith('.obj')]
            ids = [dir.split('/')[-3] for dir in obj_files]
            objs = load_objs_as_meshes(obj_files, device=self.device, 
                                       create_texture_atlas=True,
                                       load_textures=True)
            data[synset_id] = {id: obj for id, obj in zip(ids, objs)}

            meta_files = [f"{f[:-4]}.json" for f in obj_files]
            for meta_file in meta_files:
                with open(meta_file, 'r') as f:
                    object_meta = json.load(f)
                    meta[synset_id][object_meta['id']] = object_meta 
                    # object_meta['centroid'] = torch.tensor(object_meta['centroid'])
                    # object_meta['max'] = torch.tensor(object_meta['max'])
                    # object_meta['min'] = torch.tensor(object_meta['min'])
        return data, meta

    def __len__(self):
        return len(self.split_ids[self.split])

    def __getitem__(self, idx):
        synset_id, file_id = self.split_ids[self.split][idx]
        mesh = self.data[synset_id][file_id]
        meta_data = self.meta[synset_id][file_id]
        return mesh, synset_id, file_id, meta_data



class ObjRenderer:
    def __init__(self, image_size=256, device='cpu'):
        self.device = device
        self.image_size = image_size

    def __call__(self, 
                 meshes,
                 R=None, T=None):
        num_views = len(meshes)
        if R is None or T is None:
            R, T = self.sample_camera_viewing_transforms(num_views)
        
        meshes = meshes.to(self.device)
        R = R.to(self.device)
        T = T.to(self.device)
        
        # Define the settings for rasterization and shading
        raster_settings = RasterizationSettings(
            image_size=self.image_size,
            blur_radius=0.0,
            faces_per_pixel=1,
        )

        # Initialize an OpenGL perspective camera
        cameras = FoVPerspectiveCameras(device=self.device, 
                                        R=R, T=T,
                                        znear=0.1, zfar=10.0)
        
        # Create a phong renderer by composing a rasterizer and a shader
        renderer = MeshRendererWithFragments(
            rasterizer=MeshRasterizer(cameras=cameras, 
                                    raster_settings=raster_settings),
            shader=SoftPhongShader(device=self.device, cameras=cameras)
        )

        # Extract Depth
        _, fragments = renderer(meshes)
        depth_images = fragments.zbuf[:,:,:, :1]

        # Binary mask
        mask_images = (depth_images.permute(0,3,1,2) > 0)   # (b, 1, h, w)

        # Get 3D coordinates
        ## Get ij frame coordinates
        b, h, w = depth_images.shape[0:3]
        i, j = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij') 
        center = torch.tensor([h/2.0, w/2.0]).view(2, 1, 1).to(self.device)
        ji = torch.stack([j, i], dim=0).to(self.device).float()
        
        ## Get normalized coordinates (check pytorch3d docs)
        ndc = (ji / center) - 1.0
        ndc = ndc[None].repeat(num_views, 1, 1, 1).permute(0, 2, 3, 1)
        ndc = torch.concatenate([ndc, depth_images.clone()], dim=-1)
        
        ## Transform NDC to world
        # TODO: derive camera_frame coords alongside world.
        world_coordinates = cameras.unproject_points(ndc.view(b,-1, 3))
        world_coordinates = world_coordinates.view(b, h, w, 3)

        # Normalize NOCS images
        nocs_images = world_coordinates.permute(0, 3, 1, 2) + 0.5  # (b, 3, h, w)

        # Set NOCS background to 1.0
        nocs_images[mask_images.expand(b,3,h,w) == 0] = 1.0

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # TODO: Figure out why there remains some out unnormalized points
        # Note: The validation below shows some <0.0 and >1.0 values
        nocs_images = torch.clamp(nocs_images, 0.0, 1.0)
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        # Permute and mask depth
        depth_images = depth_images.permute(0,3,1,2) # (b, 1, h, w)
        depth_images = depth_images * mask_images

        return {"nocs_images":  nocs_images, 
                "depth_images": depth_images,
                "mask_images":  mask_images}


    def generate_random_camera_viewing_angles(self, count=1):
        dist = torch.rand(count) * 1.5 + 2  # Distance in meters
        elev = torch.rand(count) * 180 - 90  # Elevation in degrees
        azim = torch.rand(count) * 360       # Azimuth in degrees
        return dist, elev, azim


    def sample_camera_viewing_transforms(self, count):

        # Generate random camera views
        d, e, a = self.generate_random_camera_viewing_angles(count)
        R, T = look_at_view_transform(dist=d, elev=e, azim=a,
                                      device=self.device)
        return R, T


class NocsDataset:
    def __init__(self, 
                 dataset,
                 renderer,
                 batch_size=1):
        self.dataset = dataset
        self.renderer = renderer
        self.batch_size = batch_size

    @staticmethod
    def build(root_dir, 
              synset_ids, 
              split='train', 
              split_percentages=DEFAULT_SPLIT_PERCENTAGES,
              device='cuda', 
              verbose=False,
              batch_size=1):
        dataset = ShapeNetDataset(root_dir, 
                                  synset_ids, 
                                  split,
                                  split_percentages,
                                  verbose,
                                )
        renderer = ObjRenderer(256,
                               device)
        return NocsDataset(dataset, 
                           renderer,
                           batch_size)

    def __len__(self):
        return len(self.dataset)
    
    def __iter__(self):
        return self

    def __call__(self, idxs=None):
        return self.__getitem__(idxs)

    def __getitem__(self, idxs =None, *_, **__):
        if idxs is None:
            idxs = (torch.rand(self.batch_size) - 1e-6) * self.__len__()
        if self.batch_size > 1:
            idxs = idxs.long()
            batch = [self.dataset[i] for i in idxs]
            meshes, synset_id, file_id, meta = self.collate_fn(batch)
        else:
            meshes, synset_id, file_id, meta = self.dataset[idxs]

        renders = self.renderer(meshes)
        renders.update({"synset_id": synset_id, "file_id": file_id, "meta":meta})
        return renders

    @staticmethod
    def collate_fn(batch):
        meshes, synset_ids, file_ids, metas = zip(*batch)
        meshes = join_meshes_as_batch(meshes)
        return meshes, synset_ids, file_ids, metas


# Example usage
root_dir = '/home/baldeeb/Data/ShapeNetCore'
synset_ids = ['03797390']
NUM_VIEWS=10
dataset = ShapeNetDataset(root_dir, synset_ids, split='train', verbose=True)
renderer = ObjRenderer(256, "cuda")

# # Get depth and mask for the first mesh in the dataset
# meshes, metas = [], []
# for _ in range(NUM_VIEWS): 
#     mesh, synset_id, file_name, meta = dataset[1]
#     meshes.append(mesh)
#     metas.append(meta)
# meshes = join_meshes_as_batch(meshes)
# nocs_centroid = torch.FloatTensor([meta['centroid'] for meta in metas]).to("cuda")


# Render
# renders = renderer(meshes)
# nocs_images = renders["nocs_images"]

nocs_dataset = NocsDataset(dataset, renderer, batch_size=NUM_VIEWS)
renders = nocs_dataset()

# Validate nocs
bool_mask = renders['mask_images'].expand(-1,3,-1,-1) == 1
nocs_pts = renders["nocs_images"][bool_mask].view(-1, 3)
nocs_min, nocs_max = nocs_pts.min(0).values, nocs_pts.max(0).values
print(f"Nocs images extremes: \n\tmins: {nocs_min}\n\tmaxs: {nocs_max}")

assert torch.all(nocs_pts.min(0).values >= 0.0), "Some nocs values are smaller than 0"
assert torch.all(nocs_pts.max(0).values <= 1.0), "Some nocs values are larger than 1"

# Display the images
import matplotlib.pyplot as plt

def show_image_grid(images, title, num_cols=3):
    num_rows = int(len(images)/3)
    fig, axes = plt.subplots(num_rows, num_cols)  # Adjust figsize to your preference
    fig.suptitle(title)
    for i in range(num_rows):
        for j in range(num_cols):
            axes[i, j].imshow(images[i*num_cols + j])
            axes[i, j].axis('off')
    return fig, axes

fa1 = show_image_grid(renders["nocs_images"].permute(0,2,3,1).cpu().numpy(), "NOCS Images")
fa2 = show_image_grid(renders["depth_images"].permute(0,2,3,1).cpu().numpy(), "Depth Images")
fa3 = show_image_grid(renders["mask_images"].permute(0,2,3,1).cpu().numpy(), "Mask Images")

plt.show()
pass

