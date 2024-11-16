import torch
from .vae import VAE
from .blocks import PointNetEncoder
import wandb
import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler, UNet2DConditionModel

class PointConditionedMaskDiffuser(torch.nn.Module):
    '''This class packages the huggingface diffusion utils to facilitate use.'''
    def __init__(self, diffusion_model:UNet2DConditionModel, context_model:PointNetEncoder, scheduler:DDIMScheduler):
        super().__init__()
        self.ctxt_net = context_model
        self.diff_net = diffusion_model
        self.scheduler = scheduler
        self.num_ts = len(scheduler.betas)  # num_train_timesteps

    @property
    def encoder(self): return self.ctxt_net
    
    @property
    def diffuser(self): return self.diff_net

    def sample(self, num_inference_steps=None, **data):
        noisy = data.get('noisy', self.add_noise(data['masks']))
        return self.fix(num_inference_steps, **data, noisy=noisy)
        
    def fix(self, num_inference_steps=None, **data):
        noisy = data['noisy'].clone().detach()
        points = data['face_points']
        
        if num_inference_steps: 
            prev_num_steps = len(self.scheduler.timesteps)
            self.scheduler.set_timesteps(num_inference_steps)

        ctxt = self.ctxt_net(points).mu
        for t in self.scheduler.timesteps:
            with torch.no_grad():
                res = self.diff_net(noisy, t, ctxt).sample
            noisy = self.scheduler.step(res, t, noisy).prev_sample

        if num_inference_steps: 
            self.scheduler.set_timesteps(prev_num_steps)

        return noisy  # after backward diffusion 
    
    def sample_timesteps(self, count, device):
        return torch.randint(0, self.num_ts, (count,), device=device, dtype=torch.int64)

    def add_noise(self, images):
        timesteps = self.sample_timesteps(len(images), images.device)
        noise = torch.randn_like(images)
        return self.scheduler.add_noise(images, noise, timesteps).clamp(-1.0, 1.0)

    def forward(self, **data):
        ctxt = self.ctxt_net(data['face_points']).mu
        return self.diff_net(data['noisy'], data['timesteps'], ctxt).sample
        
    def loss(self, timesteps=None, **data):
        images = data['masks']
        if timesteps is None:
            timesteps = self.sample_timesteps(len(images), images.device)
        noise = torch.randn_like(images)
        noisy = self.scheduler.add_noise(images, noise, timesteps)
        pred = self.forward(noisy=noisy, timesteps=timesteps, **data)
        loss = F.mse_loss(noise, pred)
        return {'loss':loss}

        
class PointConditionedMaskDiffuserValidator:
    def __init__(self, dataloader, device='cuda', num_batches=1, num_inference_steps=25):
        self.dataloader = dataloader.to(device)
        self.num_batches = num_batches
        self.num_inference_steps = num_inference_steps 
    
    def _as_np(self, x): 
        return (x/2 + 0.5).permute(0,2,3,1).detach().cpu().numpy()

    def __call__(self, model, log):
        for _ in range(self.num_batches):
            d = self.dataloader()
            pred = model.sample(**d, num_inference_steps = self.num_inference_steps)
            log({"ground_truth": wandb.Image(d['masks']),
                 "predicted": wandb.Image(pred),
                 "validation loss": model.loss(**d)})
     