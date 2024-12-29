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
    def __init__(self, root_dir, synset_ids, split='train', 
                 split_percentages=DEFAULT_SPLIT_PERCENTAGES,
                 device='cpu', verbose=False):
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
        return data, meta

    def __len__(self):
        return len(self.split_ids[self.split])

    def __getitem__(self, idx):
        synset_id, file_id = self.split_ids[self.split][idx]
        mesh = self.data[synset_id][file_id]
        meta_data = self.meta[synset_id][file_id]
        return mesh, synset_id, file_id, meta_data


def generate_random_camera_views(num_views=1):
    elev = torch.rand(num_views) * 180 - 90  # Elevation in degrees
    azim = torch.rand(num_views) * 360       # Azimuth in degrees
    return elev, azim


def render_images(meshes, image_size=256, 
                  nocs_centroid=torch.FloatTensor([0, 0, 0]),
                  nocs_span=torch.FloatTensor([1, 1, 1]),
                  device = torch.device("cuda:0" 
                                        if torch.cuda.is_available() 
                                        else "cpu")):
    meshes = meshes.to(device)
    num_views = len(meshes)

    # Generate random camera views
    elev, azim = generate_random_camera_views(num_views)

    # Define the settings for rasterization and shading
    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=0.0,
        faces_per_pixel=1,
    )

    # Initialize an OpenGL perspective camera
    R, T = look_at_view_transform(dist=1.5, elev=elev, azim=azim, device=device)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
    
    # Create a phong renderer by composing a rasterizer and a shader
    renderer = MeshRendererWithFragments(
        rasterizer=MeshRasterizer(cameras=cameras, 
                                  raster_settings=raster_settings),
        shader=SoftPhongShader(device=device, cameras=cameras)
    )

    # Extract Depth
    _, fragments = renderer(meshes)
    depth_images = fragments.zbuf[:,:,:, :1]

    # Binary mask
    mask_images = (depth_images.permute(0,3,1,2) > 0)   # (b, 1, h, w)

    # Get 3D coordinates
    ## Get ij frame coordinates
    b, h, w = depth_images.shape[0:3]
    i, j = torch.meshgrid(torch.arange(h), torch.arange(w)) 
    center = torch.tensor([h/2.0, w/2.0]).view(2, 1, 1).to(device)
    ji = torch.stack([j, i], dim=0).to(device).float()
    
    ## Get normalized coordinates (check pytorch3d docs)
    ndc = (ji / center) - 1.0
    ndc = ndc[None].repeat(5, 1, 1, 1).permute(0, 2, 3, 1)
    ndc = torch.concatenate([ndc, depth_images.clone()], dim=-1)
    
    ## Transform NDC to world
    world_coordinates = cameras.unproject_points(ndc.view(b,-1, 3))
    world_coordinates = world_coordinates.view(b, h, w, 3)

    # Normalize NOCS images
    # TODO: Verify that nocs is as expected. 
    nocs_images = world_coordinates.permute(0, 3, 1, 2)  # (b, 3, h, w)
    # nocs_images -= nocs_centroid[:, :, None, None].to(device)
    # nocs_images /= nocs_span[:, :, None, None].to(device)
    nocs_images = nocs_images * mask_images

    # Permute and mask depth
    depth_images = depth_images.permute(0,3,1,2) # (b, 1, h, w)
    depth_images = depth_images * mask_images

    return nocs_images, depth_images, mask_images


# Example usage
root_dir = '/home/baldeeb/Data/ShapeNetCore'
synset_ids = ['03797390']
dataset = ShapeNetDataset(root_dir, synset_ids, split='train', verbose=True)

NUM_VIEWS=5

# Get depth and mask for the first mesh in the dataset
meshes, metas = [], []
for _ in range(NUM_VIEWS): 
    mesh, synset_id, file_name, meta = dataset[1]
    meshes.append(mesh)
    metas.append(meta)
meshes = join_meshes_as_batch(meshes)

# TODO: perform this in the dataset class
nocs_centroid = torch.FloatTensor([meta['centroid'] for meta in metas])
meta_max = torch.FloatTensor([meta['max'] for meta in metas])
meta_min = torch.FloatTensor([meta['min'] for meta in metas])

# Render
nocs_images, depth_images, mask_images = render_images(meshes, 
                                                       nocs_centroid = nocs_centroid,
                                                       nocs_span = meta_max - meta_min) 

# Display the images
import matplotlib.pyplot as plt

for i in range(NUM_VIEWS):
    plt.figure(figsize=(10, 3))
    plt.subplot(1, 3, 1)
    plt.imshow(nocs_images[i].permute(1,2,0).cpu().numpy())
    plt.title('NOCS Image')
    plt.subplot(1, 3, 2)
    plt.imshow(depth_images[i].permute(1,2,0).cpu().numpy(), cmap='gray')
    plt.title('Depth Image')
    plt.subplot(1, 3, 3)
    plt.imshow(mask_images[i].permute(1,2,0).cpu().numpy(), cmap='gray')
    plt.title('Mask Image')
    plt.show()
    pass

