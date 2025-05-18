import matplotlib.pyplot as plt

import numpy as np

import torch
from pytorch3d.renderer.cameras import look_at_view_transform

from nocs_diffusion.dataset.utils.mesh_renderer.renderer import MeshRenderer
from nocs_diffusion.dataset.utils.object_handler import ObjHandler


input_file =  "/home/baldeeb/Code/nocs-diffusion/data/nocsObjModels/00000000/1a680e3308f2aac544b2fa2cac0778f5/model.obj"
output_file = "/home/baldeeb/Code/nocs-diffusion/data/sandbox/nocs_colored.obj"

display_angles = 20
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load the object, colorize it, and convert to pytorch3d mesh
nocs_obj_loader = ObjHandler(input_file)
nocs_obj_loader.colorize_nocs()
meshes = nocs_obj_loader.get_pytorch3d_mesh(broadcast=display_angles).to(device)

# Sample camera viewing transforms
dist = [1.0 for _ in range(display_angles)]
elev = [45 for _ in range(display_angles)]
azim = [a for a in np.linspace(0, 360, display_angles)]
Rs, Ts = look_at_view_transform(dist=dist, elev=elev, azim=azim,
                                device=device)

# Get mesh renders       
renderer = MeshRenderer(image_size=512)
renders = renderer(meshes, R=Rs, T=Ts)

# Display all display angles
fig, axes = plt.subplots(2, display_angles//2, figsize=(15, 6))
for i, ax in enumerate(axes.flat):
    ax.imshow(renders['nocs_images'][i].cpu().numpy())
    ax.axis('off')
plt.tight_layout()
plt.show()
