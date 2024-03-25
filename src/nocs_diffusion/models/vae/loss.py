from collections import OrderedDict
import numpy as np

import torch
import torch.nn as nn

import torch.nn.functional as F

def vae_loss(x, x_recon, mu, log_std): 
    """
    Loss function for VAE
    Input: 
    - x: original image
    - x_recon: reconstructed image from VAE model
    - mu: mean vector of approximate posterior
    - log_std: variance vector in log space. 

    output: 
    - OrderedDict of the total loss, reconstruction loss and KL loss. 

    ------- Instruction -------
    Average the reconstruction loss and KL loss
    over batch dimension and sum over the feature dimension
    ---------------------------
    """
    x = 2*x - 1
    b = x.shape[0]

    recon_loss = F.mse_loss(x, x_recon, reduction='none').contiguous()
    recon_loss = recon_loss.view(b, -1).sum(-1).mean()

    kl_loss = -0.5 * (1 + 2*log_std - mu.square() - (2*log_std).exp())
    kl_loss = kl_loss.view(b, -1).sum(-1).mean()
    
    return OrderedDict(loss=recon_loss + kl_loss, recon_loss=recon_loss,
                        kl_loss=kl_loss)