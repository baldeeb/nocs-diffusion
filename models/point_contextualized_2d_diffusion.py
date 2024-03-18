# TODO: move to a location that houses models specific to this setup.

import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler

class PointContextualized2dDiffusionModel(torch.nn.Module):
    '''This class packages the huggingface diffusion utils to facilitate use.'''
    def __init__(self, conditioned_model, ctxt_encoder, ddpm_scheduler:DDPMScheduler):
        super().__init__()
        self.ctxt_net = ctxt_encoder
        self.diff_net = conditioned_model
        self.scheduler = ddpm_scheduler
        self.num_ts = len(ddpm_scheduler.betas)  # num_train_timesteps
    
    def fix(self, **data):
        noisy = data['noisy_images'].clone().detach()
        points = data['face_points']

        ctxt = self.ctxt_net(points).mu[:, None]
        for t in self.scheduler.timesteps:
            with torch.no_grad():
                res = self.diff_net(noisy, t, ctxt).sample
            noisy = self.scheduler.step(res, t, noisy).prev_sample
        return noisy  # after backward diffusion 
    
    def sample_timesteps(self, count, device):
        return torch.randint(0, self.num_ts, (count,), device=device, dtype=torch.int64)

    def add_noise(self, images):
        timesteps = self.sample_timesteps(len(images), images.device)
        noise = torch.randn_like(images)
        return self.scheduler.add_noise(images, noise, timesteps)

    def forward(self, **data):
        ctxt = self.ctxt_net(data['face_points']).mu[:, None]
        return self.diff_net(data['noisy_images'], data['timesteps'], ctxt).sample
        
    def loss(self, **data):
        timesteps = torch.randint(0, self.num_ts, (data['images'].shape[0],), 
                                    device=data['images'].device, dtype=torch.int64)
        noise = torch.randn_like(data['images'])
        noised = self.scheduler.add_noise(data['images'], noise, timesteps)
        pred = self.forward(noisy_images=noised, timesteps=timesteps, **data)
        loss = F.mse_loss(noise, pred)
        return {'loss':loss}
