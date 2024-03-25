from collections import OrderedDict
import numpy as np

import torch
import torch.nn as nn

import torch.nn.functional as F

from models.vae import vae_loss, ConvDecoder
from models.blocks.pointnet import PointNetEncoder

class VAEPointNetEncoder(nn.Module): 
    def __init__(self, in_dim, latent_dim, out_dim, im_size,
                 load_path=None):
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
        self.out_shape = (out_dim, im_size, im_size)
        self.latent_size = latent_dim
        self.encoder = PointNetEncoder(input_dim=in_dim, zdim=latent_dim)
        self.decoder = ConvDecoder(latent_dim, self.out_shape)
        if load_path: self.load_state_dict(torch.load(load_path))


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

    def loss(self, x, y):
        y_hat, z_mu, z_log_std = self.forward(x)
        return vae_loss(y, y_hat, z_mu, z_log_std)