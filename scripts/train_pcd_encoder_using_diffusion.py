import torch
from nocs_diffusion import train
from diffusers.optimization import get_cosine_schedule_with_warmup

import hydra
from omegaconf import DictConfig

@hydra.main(version_base=None, config_path='../config', config_name='train_pcd_encoder')
def run(cfg: DictConfig) -> None:
    model = hydra.utils.instantiate(cfg.model).to(cfg.device)
    dataloader = hydra.utils.instantiate(cfg.dataloader).to(cfg.device)
    
    validator = None
    if (cfg.validate):
        validator = hydra.utils.instantiate(cfg.validate)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=cfg.lr_warmup_steps,
        num_training_steps=(cfg.num_steps),
    )
    train(cfg, model, optimizer, lr_scheduler, dataloader, validator)

if __name__ == '__main__':
    run()