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
                                   im_size=cfg.dataset.image_size,
                                ).to(cfg.device)
        model.load_state_dict(torch.load(load_path))
        
        class encoder:
            def __init__(self, enc):
                self.enc = enc
            def __call__(self, *args, **kwargs):
                return model.encoder(*args, **kwargs)[0]

        return encoder(model.encoder)

def train(config):

    renderer = hydra.utils.instantiate(config.dataset)

    # Setup Forward Diffusion
    scheduler = DDPMScheduler()
    ctxt_encoder = get_depth_encoder(config, 
        '/home/baldeeb/Code/nocs-diffusion/checkpoints/nocs-diffusion/depth_vae/2024-03-07_17-49-55/0000_009999.pth')
    
    cfg_dict = OmegaConf.to_object(config)
    # model = NocsDiff(3, ctx_net, 64).to(config.device)  # MY OWN
    # model = get_unet(cfg_dict).to(config.device)
    model = get_conditioned_unet(cfg_dict).to(config.device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=(config.num_epochs),
    )
    scheduler.set_timesteps(config.num_train_timesteps)

    for epoch in range(config.num_epochs):

        renders = renderer()
        images = renders['images'].permute(0, 3, 1, 2).to(config.device)
        pts = sample_from_clouds(renders['face_points'], 1000).to(config.device)



        # forward diffusion
        timesteps = torch.randint(0, config.num_train_timesteps, (images.shape[0],), 
                                device=config.device, dtype=torch.int64)
        noise = torch.randn_like(images)
        noised_images = scheduler.add_noise(images, noise, timesteps)
        # viz_image_batch(noised_images.permute(0,2,3,1).detach().cpu().numpy())

        # pred_noise = model(noised_images, depths)  # MY OWN

        # pred_noise = model(noised_images, timesteps).sample  # Unconditioned UNET
        

        # TODO: still has some shape issue....
        ctxt = ctxt_encoder(pts)[:, None]
        # ctxt = torch.zeros(18, 1, 1028).to(config.device)
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
    train(cfg)

if __name__ == '__main__':
    run()