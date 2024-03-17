import torch
import torch.nn.functional as F

from models.unets import get_conditioned_unet
from utils.visualization import viz_image_batch

from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers import DDPMScheduler

from omegaconf import DictConfig, OmegaConf
import hydra
import os

from utils.train import train as train_util

def visualize(dataloader, model, ddpm_scheduler):
    as_np = lambda x: (x/2 + 0.5).permute(0,2,3,1).detach().cpu().numpy()
    data = dataloader()
    images = data['images']
    viz_image_batch(as_np(images), block=False, title='Original')
    
    timesteps = model.sample_timesteps(len(images), images.device)
    noised_images = ddpm_scheduler.add_noise(images, torch.randn_like(images), timesteps)
    viz_image_batch(as_np(noised_images), block=False, title='Noised')
    
    fixed_images = model.fix(noisy_images=noised_images, **data)
    viz_image_batch(as_np(fixed_images), title='Fixed')


class UnetWrapper(torch.nn.Module):
    def __init__(self, ctxt_encoder, conditioned_unet, ddpm_scheduler):
        super().__init__()
        self.ctxt_net = ctxt_encoder
        self.unet = conditioned_unet
        self.scheduler = ddpm_scheduler
        self.num_ts = len(ddpm_scheduler.betas)  # num_train_timesteps
    
    def fix(self, **data):
        noisy = data['noisy_images'].clone().detach()
        points = data['face_points']

        ctxt = self.ctxt_net(points).mu[:, None]
        for t in self.scheduler.timesteps:
            with torch.no_grad():
                res = self.unet(noisy, t, ctxt).sample  # Conditioned UNet
            noisy = self.scheduler.step(res, t, noisy).prev_sample
        return noisy  # after backward diffusion 
    
    def sample_timesteps(self, count, device):
        return torch.randint(0, self.num_ts, (count,), device=device, dtype=torch.int64)

    def forward(self, **data):
        ctxt = self.ctxt_net(data['face_points']).mu[:, None]
        return self.unet(data['noisy_images'], data['timesteps'], ctxt).sample
        
    def loss(self, **data):
        timesteps = torch.randint(0, self.num_ts, (data['images'].shape[0],), 
                                    device=data['images'].device, dtype=torch.int64)
        noise = torch.randn_like(data['images'])
        noised = self.scheduler.add_noise(data['images'], noise, timesteps)
        pred = self.forward(noisy_images=noised, timesteps=timesteps, **data)
        loss = F.mse_loss(noise, pred)
        return {'loss':loss}



@hydra.main(version_base=None, config_path='./config', config_name='diffuser')
def run(cfg: DictConfig) -> None:

    if cfg.log_locally:
        os.environ["WANDB_MODE"] = "offline"
    
    dataloader = hydra.utils.instantiate(cfg.dataloader).to(cfg.device)

    # Setup Forward Diffusion
    ddpm_scheduler = DDPMScheduler()
    ddpm_scheduler.set_timesteps(cfg.num_train_timesteps)
    
    # Load context encoder
    load_path = '/home/baldeeb/Code/nocs-diffusion/checkpoints/nocs-diffusion/depth_vae/2024-03-15_21-40-45/0000_000000.pth'
    vae = hydra.utils.instantiate(cfg.vae).to(cfg.device)
    if load_path is not None: vae.load_state_dict(torch.load(load_path))
    ctxt_encoder = vae.encoder
    
    # Instantiate diffusion model
    cfg_dict = OmegaConf.to_object(cfg)
    unet = get_conditioned_unet(cfg_dict).to(cfg.device)
    model = UnetWrapper(ctxt_encoder, unet, ddpm_scheduler)

    # Set up optimizer  TODO: move to hydra config.
    optimizer = torch.optim.AdamW(unet.parameters(), lr=cfg.lr)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=cfg.lr_warmup_steps,
        num_training_steps=(cfg.num_epochs),
    )

    # Run training
    train_util(cfg, model, optimizer, lr_scheduler, dataloader)
    visualize(dataloader, model, ddpm_scheduler)

if __name__ == '__main__':
    run()