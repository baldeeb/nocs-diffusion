from omegaconf import DictConfig, OmegaConf
import wandb
import hydra
import os
from tqdm import tqdm
import pathlib as pl

from utils.load_save import save_model


def train(config, model, optimizer, lr_scheduler, dataloader):

    # Logger
    if config.log: 
        wandb.init(**config.logger, config=OmegaConf.to_container(config))
        log = wandb.log
    else: log = lambda x: None

    epoch = 0
    batch_tqdm = tqdm(range(config.num_epochs), 
                      desc='Training Step Loop')
    for batch_i in batch_tqdm:
        log({'step': batch_i+1})

        data = dataloader()
        loss = model.loss(**data)  # TODO: remove this train fx or make loss fx more versatile.
        log(loss)
        
        optimizer.zero_grad()
        loss['loss'].backward()
        optimizer.step()
        lr_scheduler.step()

        if config.steps_before_save and (batch_i % config.steps_before_save) == 0:
            batch_tqdm.set_description(f'Saving batch {batch_i}, loss {loss["loss"]:.2f}')
            save_model(model, pl.Path(config.checkpoint_dir), epoch, batch_i,
                        retain_n=config.get('retain_n_checkpoints', None))

    save_model(model, pl.Path(config.checkpoint_dir), epoch, batch_i,
                retain_n=config.get('retain_n_checkpoints', None))
