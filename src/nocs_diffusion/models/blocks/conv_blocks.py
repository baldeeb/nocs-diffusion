import numpy as np
import torch.nn as nn


'''
This class upsizes by linearly projecting an image to a certain shape 
    then doubling for len(layer_dims - 1) times.
'''
class ConvDecoder(nn.Module):
    def __init__(self, in_dim, layer_dims, out_dim, out_image_size):
        super().__init__()
        assert len(layer_dims) > 2
        self.in_dim = in_dim  # dim of the embedding
        self.layer_dims = layer_dims
        self.out_dim = out_dim
        self.image_size = out_image_size

        # shape of the image linearly projected from latent before deconv. 
        base_im_size = out_image_size // 2**(len(layer_dims) - 1)
        self.base_shape = (self.layer_dims[0], base_im_size, base_im_size)
        
        # layer projecting latent to image
        self.fc = nn.Linear(in_dim, np.prod(self.base_shape))

        # deconv layers
        self.deconvs = []
        for l_in, l_out in zip(self.layer_dims[:-1], self.layer_dims[1:]):
            self.deconvs.extend([
                nn.ReLU(),
                nn.ConvTranspose2d(l_in, l_out, 4, stride=2, padding=1),
            ])
        self.deconvs.extend([
            nn.ReLU(),
            nn.Conv2d(layer_dims[-1], out_dim, 3, padding=1)
        ])
        self.deconvs = nn.Sequential(*self.deconvs)

    def forward(self, z):
        out = self.fc(z).contiguous()
        out = out.view(out.shape[0], *self.base_shape)
        out = self.deconvs(out)
        return out

class ConvEncoder(nn.Module):
    def __init__(self, layer_dims, in_image_size):
        super().__init__()
        self.image_size = in_image_size
        self.layer_dims = layer_dims
        
        self.convs = []
        for l0, l1 in zip(layer_dims[:-1], layer_dims[1:]):
            self.convs.extend([
                nn.Conv2d(l0, l1, 3, padding=1),
                nn.ReLU()
            ])
        self.convs = nn.Sequential(*self.convs)
        conv_out_dim = layer_dims * (in_image_size // 8 )**2 
        self.fc = nn.Linear(conv_out_dim, 2 * layer_dims[-1])

    def forward(self, x):
        out = self.convs(x).contiguous()
        out = out.view(out.shape[0], -1)
        mu, log_std = self.fc(out).chunk(2, dim=1)
        return mu, log_std
