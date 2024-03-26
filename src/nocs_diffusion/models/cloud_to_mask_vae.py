import torch
from .vae import VAE
from .blocks import PointNetEncoder, ConvDecoder


class VAEPointNetEncoderConvDecoder(VAE): 
    def __init__(self, in_dim, latent_dim, out_dim, im_size,
                load_path=None):
        
        out_shape = (out_dim, im_size, im_size)
        super().__init__(PointNetEncoder(input_dim=in_dim, zdim=latent_dim),
                         ConvDecoder(latent_dim, out_shape),)
        if load_path: self.load_state_dict(torch.load(load_path))

class CloudToMaskVae(torch.nn.Module):
        def __init__(self, model:VAEPointNetEncoderConvDecoder):
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