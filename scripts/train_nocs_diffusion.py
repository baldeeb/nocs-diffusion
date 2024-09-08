import torch
from diffusers.optimization import get_cosine_schedule_with_warmup
from nocs_diffusion import train, ConfigDirectoriesManager, ConfigLoader
from nocs_diffusion.utils.load_save import save_model

from omegaconf import OmegaConf
import wandb
import os
from tqdm import tqdm
import pathlib as pl


if __name__ == '__main__':
    cfg_path = ConfigDirectoriesManager()['train_diffusion.yaml']
    loader = ConfigLoader.from_config_path(str(cfg_path))
    cfg = loader.cfg
    dataloader = loader.dataloader
    model = loader.model
    
    # TODO: move optimizer to hydra config.
    optimizer = torch.optim.AdamW(model.diff_net.parameters(), lr=loader.cfg.lr)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=loader.cfg.lr_warmup_steps,
        num_training_steps=(loader.cfg.num_steps),
    )
    train(loader.cfg, model, optimizer, lr_scheduler, dataloader)
    
    # Logger
    if cfg.log: 
        wandb.init(**cfg.logger, config=OmegaConf.to_container(cfg))
        log = wandb.log
        if cfg.log_locally: os.environ["WANDB_MODE"] = "offline"
    else: log = lambda _: None

    checkpoint_i = 0
    batch_tqdm = tqdm(range(cfg.num_steps), 
                      desc='Training Step Loop')
    for batch_i in batch_tqdm:
        log({'step': batch_i+1})

        data = dataloader()
        loss = model.loss(**data)
        log(loss)
        
        optimizer.zero_grad()
        loss['loss'].backward()
        optimizer.step()
        lr_scheduler.step()

        if cfg.steps_before_save and (batch_i % cfg.steps_before_save) == 0:
            batch_tqdm.set_description(f'Saving batch {batch_i}, loss {loss["loss"]:.2f}')
            save_model(model, pl.Path(cfg.checkpoint_dir), checkpoint_i, batch_i,
                        retain_n=cfg.get('retain_n_checkpoints', None))
        
        # if cfg.steps_before_eval and (batch_i % cfg.steps_before_save) == 0:
        #     eval

    save_model(model, pl.Path(cfg.checkpoint_dir), checkpoint_i, batch_i,
                retain_n=cfg.get('retain_n_checkpoints', None))

