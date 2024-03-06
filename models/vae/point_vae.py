from collections import OrderedDict
import numpy as np

import torch
import torch.nn as nn

import torch.nn.functional as F

from models.vae import vae_loss, ConvDecoder
from models.pointnet import PointNetEncoder

class VAEPointNetEncoder(nn.Module): 
    def __init__(self, input_shape, latent_size):
        """
        Simple VAE model. 
        
        Inputs: 
        - input_shape: size of input with shape (C, H, W)
        - latent_size: size of latent variable

        ------- Instruction -------
        You could follow the recommended network architecture in vae.ipynb
        
        ---------------------------
        """
        super().__init__()
        assert len(input_shape) == 3

        self.input_shape = input_shape
        self.latent_size = latent_size
        
        self.encoder = PointNetEncoder(input_dim=3, zdim=latent_size)
        self.decoder = ConvDecoder(latent_size, input_shape)

    def forward(self, x):
        # TODO: finish the forward pass of VAE model
        # Normalize the input to [-1, 1]
        
        x = 2*x - 1
        (mu, log_std) = self.encoder(x)
        
        zeros = torch.zeros_like(mu)
        ones = torch.ones_like(log_std)
        z = mu + torch.normal(zeros, ones) * log_std.exp()

        # x_recon =  torch.clamp(self.decoder(z), -1, 1)*0.5 + 0.5
        x_recon =  torch.clamp(self.decoder(z), -1, 1)
        return x_recon, mu, log_std

    def sample(self, n):
        with torch.no_grad():
            z = torch.randn(n, self.latent_size).cuda()
            samples = torch.clamp(self.decoder(z), -1, 1)
        return samples.cpu().numpy() * 0.5 + 0.5
