import torch
from diffusers.optimization import get_cosine_schedule_with_warmup
from nocs_diffusion import train, ConfigDirectoriesManager, ConfigLoader
from nocs_diffusion.utils.load_save import save_model

import hydra
from omegaconf import DictConfig, OmegaConf

import wandb
import os
from tqdm import tqdm
import pathlib as pl


@hydra.main(version_base=None, config_path='../src/nocs_diffusion/config', config_name='train_diffusion')
def run(cfg: DictConfig) -> None:
    model = hydra.utils.instantiate(cfg.model).to(cfg.device)
    dataloader = hydra.utils.instantiate(cfg.dataloader).to(cfg.device)
    
    # TODO: move optimizer to hydra config.
    optimizer = torch.optim.AdamW(model.diff_net.parameters(), lr=cfg.lr)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=cfg.lr_warmup_steps,
        num_training_steps=(cfg.num_steps),
    )

    train(cfg, model, optimizer, lr_scheduler, dataloader)


if __name__ == '__main__':
    run()