import torch
import torch.nn as nn

from . import vae_loss


class VAE(nn.Module): 
    def __init__(self, encoder, decoder):
        """
        Simple VAE model. 
        
        Inputs: 
        - input_shape: size of input with shape (C, H, W)
        - latent_size: size of latent variable
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder


    def forward(self, x):
        # Normalize the input to [-1, 1]
        x = 2*x - 1
        (mu, log_std) = self.encoder(x)
        zeros = torch.zeros_like(mu)
        ones = torch.ones_like(log_std)
        z = mu + torch.normal(zeros, ones) * log_std.exp()
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
    
