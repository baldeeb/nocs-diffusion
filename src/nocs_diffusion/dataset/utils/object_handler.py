import numpy as np
import os
import sys

from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesVertex
from pytorch3d.io import load_obj

import torch

class ObjHandler:
    """ Currently assumes that the obj file has no texture and no vertex colors."""

    def __init__(self, input_file):
        self.obj = self._load_obj(input_file) 

    def _load_obj(self, input_file):
        with open(input_file, 'r') as file:
            return load_obj(file, load_textures=False)

    def get_pytorch3d_mesh(self, broadcast=1):
        """ Convert the vertices and faces to a PyTorch3D Meshes object.
        Args:
            broadcast (int): The number of times to repeat the mesh in the batch dimension.
            This allows for batching the same object without copying the data. Something not
            supported by Pytorch3d's Meshes class. 
        Returns:
            mesh (Meshes): A PyTorch3D Meshes object with the vertices, faces, and colors.
        """
        # Create a TexturesVertex object
        textures = TexturesVertex(
            verts_features=self.colors.expand(broadcast, -1, -1)) # Add batch dimension

        # Create the Meshes object
        mesh = Meshes(
            verts=self.obj[0].expand(broadcast, -1, -1),
            faces=self.obj[1].verts_idx.expand(broadcast, -1, -1),
            textures=textures)
        
        return mesh

    def colorize_nocs(self):
        """ Colorize the object with NOCS colors."""
        # Normalize the vertices to the range [0, 1]
        min_vals = self.obj[0].min(0).values
        max_vals = self.obj[0].max(0).values
        self.colors = (self.obj[0] - min_vals) / (max_vals - min_vals)

    def colorize(self, colors: torch.Tensor):
        """ Colorize the object with the given colors.
        Args:
            colors (torch.Tensor): A tensor of shape (N, 3) representing the RGB colors.
        """
        if colors.shape[1] != 3:
            raise ValueError("Colors must be of shape (N, 3)")
        self.colors = colors

    def save_obj(self, output_path):
        colored_vertices = np.concatenate((self.obj[0].cpu().numpy(), self.colors.cpu().numpy()), axis=1)
        with open(output_path, 'w') as file:
            for vertex in colored_vertices:
                file.write(f"v {' '.join(map(str, vertex))}\n")
            for face in self.obj[1].verts_idx.cpu().numpy():
                file.write(f"f {' '.join(str(idx + 1) for idx in face)}\n")



if __name__ == "__main__":
    # if len(sys.argv) != 3:
    #     print("Usage: python main.py <input_obj_file> <output_obj_file>")
    #     sys.exit(1)

    # input_file = sys.argv[1] # /home/baldeeb/Code/nocs-diffusion/data/nocsObjModels/00000000/1a680e3308f2aac544b2fa2cac0778f5/model.mtl
    # output_file = sys.argv[2] # /home/baldeeb/Code/nocs-diffusion/data/sandbox/nocs_colored.obj
    input_file =  "/home/baldeeb/Code/nocs-diffusion/data/nocsObjModels/00000000/1a680e3308f2aac544b2fa2cac0778f5/model.obj"
    output_file = "/home/baldeeb/Code/nocs-diffusion/data/sandbox/nocs_colored.obj"

    if not os.path.exists(input_file):
        print(f"Error: The file {input_file} does not exist.")
        sys.exit(1)

    oh = ObjHandler(input_file)
    meshes = oh.get_pytorch3d_mesh()
    oh.colorize_nocs()
    print(f"Mesh vertices: {meshes.verts_packed()}")
    print(f"Mesh faces: {meshes.faces_packed()}")
    print(f"Mesh colors: {meshes.textures.verts_features_packed()}")
    oh.save_obj(output_file)

    print(f"Processed {input_file} and saved to {output_file}.")