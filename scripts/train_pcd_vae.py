import torch
from nocs_diffusion import train, ConfigDirectoriesManager, ConfigLoader
from diffusers.optimization import get_cosine_schedule_with_warmup

import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(version_base=None, config_path='../src/nocs_diffusion/config', config_name='train_depth_vae')
def run(cfg: DictConfig) -> None:
    model = hydra.utils.instantiate(cfg.model).to(cfg.device)
    dataloader = hydra.utils.instantiate(cfg.dataloader).to(cfg.device)
    
    eval = None
    if (cfg.eval):
        eval = hydra.utils.instantiate(cfg.eval)

    # TODO: move optimizer to hydra config.
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=cfg.lr_warmup_steps,
        num_training_steps=(cfg.num_steps),
    )
    train(cfg, model, optimizer, lr_scheduler, dataloader, eval)

if __name__ == '__main__':
    run()