import torch

from dataset import ShapeNetRenderer, sample_from_clouds, mask_from_depth, add_gaussian_noise, sample_transforms

from utils.visualization import viz_image_batch
from utils.load_save import save_model

from diffusers.optimization import get_cosine_schedule_with_warmup

from models.vae import VAEPointNetEncoder

from omegaconf import DictConfig, OmegaConf
import wandb
import hydra
import os

from utils.common import train



@hydra.main(version_base=None, config_path='./config', config_name='depth_vae')
def run(cfg: DictConfig) -> None:
    if cfg.log_locally:
        os.environ["WANDB_MODE"] = "offline"

    model = VAEPointNetEncoder(in_dim=3, 
                               latent_dim=cfg.depth_latent_size, 
                               out_dim=1,
                               im_size=cfg.dataset.image_size
                        ).to(cfg.device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=cfg.lr_warmup_steps,
        num_training_steps=(cfg.num_epochs),
    )

    dataloader = hydra.utils.instantiate(cfg.dataloader).to(cfg.device)

    train(cfg, model, optimizer, lr_scheduler, dataloader)

if __name__ == '__main__':
    run()