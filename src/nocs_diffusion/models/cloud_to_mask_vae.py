import torch
from .vae import VAEPointNetEncoder

class CloudToMaskVae(torch.nn.Module):
        def __init__(self, model:VAEPointNetEncoder, load_path=None):
            super().__init__()
            self.net = model
            if load_path: self.load_state_dict(torch.load(load_path))


        @property
        def encoder(self): return self.net.encoder
        
        @property
        def decoder(self): return self.net.decoder

        def forward(self, **data):
            return self.net(data['face_points'])
        
        def loss(self, **data):
            return self.net.loss(data['face_points'], data['masks'])