import torch

from utils.visualization import viz_image_batch
from utils.load_save import save_model

from diffusers.optimization import get_cosine_schedule_with_warmup

from models.vae import VAEPointNetEncoder

from utils.visualization import viz_image_batch
from utils.load_save import save_model

from omegaconf import DictConfig, OmegaConf
import wandb
import hydra
import os
from tqdm import tqdm
import pathlib as pl


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
        loss = model.loss(data['face_points'], data['masks'])  # TODO: remove this train fx or make loss fx more versatile.
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


def visualize_sample(dataloader, model):
    data = dataloader()
    x, y = data['face_points'], data['masks']
    y_hat, _, _, = model(x)
    as_np = lambda x: (x/2 + 0.5).permute(0,2,3,1).detach().cpu().numpy()
    viz_image_batch(as_np(y), block=False, title='Original')
    viz_image_batch(as_np(y_hat), title='Reconstructed')


@hydra.main(version_base=None, config_path='./config', config_name='depth_vae')
def run(cfg: DictConfig) -> None:
    if cfg.log_locally:
        os.environ["WANDB_MODE"] = "offline"

    model = hydra.utils.instantiate(cfg.model).to(cfg.device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=cfg.lr_warmup_steps,
        num_training_steps=(cfg.num_epochs),
    )

    dataloader = hydra.utils.instantiate(cfg.dataloader).to(cfg.device)

    train(cfg, model, optimizer, lr_scheduler, dataloader)

    visualize_sample(dataloader, model)

if __name__ == '__main__':
    run()