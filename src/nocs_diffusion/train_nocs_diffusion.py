import torch

from utils.train import train
from diffusers.optimization import get_cosine_schedule_with_warmup

from omegaconf import DictConfig
import hydra



@hydra.main(version_base=None, config_path='./config', config_name='diffuser')
def run(cfg: DictConfig) -> None:
    dataloader = hydra.utils.instantiate(cfg.dataloader).to(cfg.device)
    model = hydra.utils.instantiate(cfg.diffusion_model).to(cfg.device)
    
    # TODO: move optimizer to hydra config.
    optimizer = torch.optim.AdamW(model.diff_net.parameters(), lr=cfg.lr)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=cfg.lr_warmup_steps,
        num_training_steps=(cfg.num_epochs),
    )
    train(cfg, model, optimizer, lr_scheduler, dataloader)

if __name__ == '__main__':
    run()