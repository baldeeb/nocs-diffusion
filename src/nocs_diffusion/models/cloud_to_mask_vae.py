import torch
from .vae import VAE
from .blocks import PointNetEncoder, ConvDecoder

# TODO: fix up this function. Currently not in use.
class VAEPointNetEncoderConvDecoder(VAE): 
    def __init__(self, in_dim, latent_dim, out_dim, im_size,
                load_path=None):
        
        super().__init__(
            PointNetEncoder(in_dim=in_dim, 
                            layer_dims=[128, 256, 512],  # TODO: move to config.
                            out_dim=latent_dim),
            ConvDecoder(in_dim=latent_dim, 
                        layer_dims=[128, 64, 32],  # TODO: move to config.
                        out_dim=out_dim, 
                        out_image_size=im_size)
        )
        if load_path: self.load_state_dict(torch.load(load_path))

class CloudToMaskVae(torch.nn.Module):
        def __init__(self, model:VAE):
            super().__init__()
            self.net = model

        @property
        def encoder(self): return self.net.encoder
        
        @property
        def decoder(self): return self.net.decoder

        def forward(self, **data):
            return self.net(data['face_points'])
        
        def loss(self, **data):
            return self.net.loss(data['face_points'], data['masks'])