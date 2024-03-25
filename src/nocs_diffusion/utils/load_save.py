import pathlib as pl
import os
import torch
import logging
from omegaconf import OmegaConf
import hydra

def load_from_config(config_path, config_name, job_name='inference', model_path=None, overrides=[]):
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize(config_path=config_path, job_name=job_name)
    cfg = hydra.compose(config_name=config_name, overrides=overrides)
    if model_path is not None: cfg.model.load = model_path
    model = hydra.utils.instantiate(cfg.model)
    model.load_state_dict(torch.load(cfg.model.load))
    logging.info(f'Loaded {cfg.model.load}')
    return model, cfg

def save_config(cfg: OmegaConf, path: pl.Path):
    assert path.suffix == '.yaml', \
        f'Config file must be a .yaml file. Got {path.suffix}'
    if not path.parent.exists(): 
        os.makedirs(path.parent)
    with open(str(path), 'w+') as f: 
        f.write(OmegaConf.to_yaml(cfg))

def save_model(model, path, epoch, batch, retain_n=None):
    path = pl.Path(path) / f'{epoch:0>4d}_{batch:0>6d}.pth'
    if not path.parent.exists():
        os.makedirs(path.parent)
    torch.save(model.state_dict(), path)
    logging.debug(f"Saved model to: {path}")
    if retain_n: 
        chkpts = [f.name for f in path.parent.glob('*.pth')]
        chkpts = sorted(chkpts, reverse=True)
        for c in chkpts[retain_n:]: os.remove(path.parent/c)
