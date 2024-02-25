
from torch import nn

from models.encoders.pointnet import PointNetEncoder
from models.diffusion import VarianceSchedule, PointwiseNet, ContextualDiffusion

class PoseDiffModel(nn.Module):
    def __init__(self):
        super().__init__()
        pt_dim = 3
        T_dim = 4 + 3 + 1  # quaternion + translation + scale
        num_steps = 100
        beta_1 = 1e-4
        beta_T = 0.02
        sched_mode = 'linear'
        latent_dim = 256
        residual = True

        self.encoder = PointNetEncoder(
            latent_dim, 
            pt_dim
        )
        self.diffusion = ContextualDiffusion(
            net = PointwiseNet(
                point_dim=T_dim, 
                context_dim=latent_dim, 
                residual=residual
            ),
            var_sched = VarianceSchedule(
                num_steps=num_steps,
                beta_1=beta_1,
                beta_T=beta_T,
                mode=sched_mode,
            )
        )

    def __call__(self, points, transform=None):
        if self.training():
            assert transform is not None, 'Must provide a transform in training mode'
            return self.get_loss(points, transform)
        else:
            return self.get_transform(points)

    def get_transform(self, points):
        '''
        Args:
            points: (B, N, D) tensor of context points 
                (typcially D=3)
        '''
        points = points.to(self.device)
        z_mu, _ = self.encoder(points)
        transform = self.diffusion.get_transform(z_mu)
        return transform

    def to(self, device):
        self.device = device
        return super().to(device)

    def get_loss(self, points, transform):
        '''
        Args:
            context: (B, N, D) tensor of context points 
                (typcially D=3)
            x: (B, 4+3+1) tensor of query SE3 transform
        '''
        # Question: Shuold the context be diffused?
        # Answer: not atm. It will stop being a normal diffusion model if the 
        #       context is changed with the diffusion.

        points = points.to(self.device)
        transform = transform.unsqueeze(1).to(self.device)
        z_mu, _ = self.encoder(points)
        loss_recons = self.diffusion.get_loss(transform, z_mu)

        return loss_recons
