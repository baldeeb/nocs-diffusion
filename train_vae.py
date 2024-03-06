import torch
import torch.nn.functional as F

from pytorch3d.ops import estimate_pointcloud_normals

from utils.renderer import RendererWrapper, sample_transforms, mask_from_depth
from utils.dataloader import PointCloudLoader
from utils.visualization import viz_image_batch

from utils.nocs_generator import nocs_extractor

from diffusers.optimization import get_cosine_schedule_with_warmup

from models.vae import VAEConvNet, vae_loss

import wandb

import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

import pathlib as pl
from utils.load_save import save_model
import os

def train(config):


    def add_noise(x, mu=0, std=0.005):
        return x + torch.randn(x.shape, device=x.device) * std + mu

    # Logger
    if config.log: 
        wandb.init(**config.logger, config=OmegaConf.to_container(config))
        log = wandb.log
    else: log = lambda x: None
    
    # Setup Dataset
    dataloader = PointCloudLoader(
                            path='./data/shapenet.hdf5',
                            categories=['chair'],
                            split='test',
                            batch_size=config.num_objects,
                            shuffle=False,
                            device=config.device, 
                            post_process=add_noise
                        )
    render = RendererWrapper(image_size=config.image_size)
    
    channel_size = 1
    input_shape = (channel_size, config.image_size, config.image_size)
    model = VAEConvNet(input_shape, config.depth_latent_size).to(config.device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=(config.num_epochs),
    )

    # Run Rendering
    pts = dataloader()
    feats = nocs_extractor(pts)
    normals = estimate_pointcloud_normals(pts).mean(0)

    # select a single object
    pts = pts[0]
    feats = (feats[0] - 0.5) * 2  # try range (-1, +1)

    epoch = 0
    for batch_i in tqdm(range(config.num_epochs), desc='Training Step Loop'):
        log({'step': batch_i+1})

        Rs, Ts = sample_transforms(config.num_views, elev_range=[0, 70])
        renders = render(pts, feats, 
                        #  normals=normals, # Renders normals if included
                         Rs=Rs, Ts=Ts)
        
        x = renders['depths'].permute(0, 3, 1, 2).to(config.device)
        # masks = mask_from_depth(x, inverse=True)
        # x = torch.concatenate([x, masks], dim=1)
        
        x_hat, z_mu, z_log_std = model(x)
        loss = vae_loss(x, x_hat, z_mu, z_log_std)

        log(loss)
        
        optimizer.zero_grad()
        loss['loss'].backward()
        optimizer.step()
        lr_scheduler.step()

        if config.steps_before_save and (batch_i % config.steps_before_save) == 0:
            save_model(model, pl.Path(config.checkpoint_dir), epoch, batch_i,
                        retain_n=config.get('retain_n_checkpoints', None))

    save_model(model, pl.Path(config.checkpoint_dir), epoch, batch_i,
                retain_n=config.get('retain_n_checkpoints', None))

                
    as_np = lambda x: (x/2 + 0.5).permute(0,2,3,1).detach().cpu().numpy()
    # if x.shape[1] == 2: x, x_hat = x[:, 0:1], x_hat[:, 0:1]
    viz_image_batch(as_np(x), block=False, title='Original')
    viz_image_batch(as_np(x_hat), title='Reconstructed')
    pass


@hydra.main(version_base=None, config_path='./config', config_name='depth_vae')
def run(cfg: DictConfig) -> None:
    if cfg.log_locally:
        os.environ["WANDB_MODE"] = "offline"
    train(cfg)

if __name__ == '__main__':
    run()