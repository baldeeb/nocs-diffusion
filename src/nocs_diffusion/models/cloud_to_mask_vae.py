import torch
from .vae import VAEPointNetEncoderConvDecoder

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