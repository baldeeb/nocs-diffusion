import torch
import torch.nn.functional as F

from models.nocs_diffusion import NocsDiff, CtxtEncoder
from models.scheduler import VarianceSchedule
from models.unets import get_unet, get_conditioned_unet

from dataset import sample_from_clouds
from utils.visualization import viz_image_batch

from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers import DDPMScheduler

from torchvision import transforms


from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
import pathlib as pl
import wandb
import hydra
import os


from models.vae import VAEPointNetEncoder
def get_depth_encoder(cfg, load_path):
        model = VAEPointNetEncoder(in_dim=3,
                                   latent_dim=cfg.depth_latent_size,
                                   out_dim=1,
                                   im_size=cfg.image_size,
                                ).to(cfg.device)
        model.load_state_dict(torch.load(load_path))
        model.encoder.mean_only = True
        return model.encoder

def train(config, model, ctxt_encoder, scheduler, optimizer, lr_scheduler, dataloader):
   

    for epoch in range(config.num_epochs):

        data = dataloader()
        images, pts = data['images'], data['face_points']


        # forward diffusion
        timesteps = torch.randint(0, config.num_train_timesteps, (images.shape[0],), 
                                device=config.device, dtype=torch.int64)
        noise = torch.randn_like(images)
        noised_images = scheduler.add_noise(images, noise, timesteps)

        # pred_noise = model(noised_images, depths)  # MY OWN
        # pred_noise = model(noised_images, timesteps).sample  # Unconditioned UNET
        ctxt = ctxt_encoder(pts)[0][:, None]
        pred_noise = model(noised_images, timesteps, ctxt).sample  # Conditioned UNET
        
        loss = F.mse_loss(noise, pred_noise)
        loss.backward()
        
        # print loss
        if epoch % 500 == 0: print(loss)
        
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()


    as_np = lambda x: (x/2 + 0.5).permute(0,2,3,1).detach().cpu().numpy()

    viz_image_batch(as_np(images), block=False, title='Original')
    viz_image_batch(as_np(noised_images), block=False, title='Noised')

    img = noised_images.clone().detach()
    c = ctxt.clone().detach()

    for t in scheduler.timesteps:
        with torch.no_grad():
            # noisy_residual = model(input, depths)  # MY OWN
            # noisy_residual = model(img, t).sample  # UNet
            noisy_residual = model(img, t, c).sample  # Conditioned UNet
        previous_noisy_sample = scheduler.step(noisy_residual, t, img).prev_sample
        img = previous_noisy_sample
    
    viz_image_batch(as_np(img), title='Fixed')
    pass


@hydra.main(version_base=None, config_path='./config', config_name='diffuser')
def run(cfg: DictConfig) -> None:

    if cfg.log_locally:
        os.environ["WANDB_MODE"] = "offline"
    
    dataloader = hydra.utils.instantiate(cfg.dataloader).to(cfg.device)

    # Setup Forward Diffusion
    scheduler = DDPMScheduler()
    ctxt_encoder = get_depth_encoder(cfg, 
        '/home/baldeeb/Code/nocs-diffusion/checkpoints/nocs-diffusion/depth_vae/2024-03-07_17-49-55/0000_009999.pth')
    
    cfg_dict = OmegaConf.to_object(cfg)
    # model = NocsDiff(3, ctx_net, 64).to(config.device)  # MY OWN
    # model = get_unet(cfg_dict).to(config.device)
    model = get_conditioned_unet(cfg_dict).to(cfg.device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=cfg.lr_warmup_steps,
        num_training_steps=(cfg.num_epochs),
    )
    scheduler.set_timesteps(cfg.num_train_timesteps)
    
    
    train(cfg, model, ctxt_encoder, scheduler, optimizer, lr_scheduler, dataloader)

if __name__ == '__main__':
    run()