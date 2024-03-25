import torch 
from .vae import VAE
from ..blocks import PointNetEncoder, ConvDecoder

class VAEPointNetEncoderConvDecoder(VAE): 
    def __init__(self, in_dim, latent_dim, out_dim, im_size,
                load_path=None):
        
        out_shape = (out_dim, im_size, im_size)
        super().__init__(PointNetEncoder(input_dim=in_dim, zdim=latent_dim),
                         ConvDecoder(latent_dim, out_shape),)
        if load_path: self.load_state_dict(torch.load(load_path))
