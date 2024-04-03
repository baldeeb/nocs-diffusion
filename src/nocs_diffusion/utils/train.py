from omegaconf import OmegaConf
import wandb
import os
from tqdm import tqdm
import pathlib as pl

from .load_save import save_model


def train(cfg, model, optimizer, lr_scheduler, dataloader):

    # Logger
    if cfg.log: 
        wandb.init(**cfg.logger, config=OmegaConf.to_container(cfg))
        log = wandb.log
        if cfg.log_locally: os.environ["WANDB_MODE"] = "offline"
    else: log = lambda _: None

    epoch = 0
    batch_tqdm = tqdm(range(cfg.num_epochs), 
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
            save_model(model, pl.Path(cfg.checkpoint_dir), epoch, batch_i,
                        retain_n=cfg.get('retain_n_checkpoints', None))

    save_model(model, pl.Path(cfg.checkpoint_dir), epoch, batch_i,
                retain_n=cfg.get('retain_n_checkpoints', None))
