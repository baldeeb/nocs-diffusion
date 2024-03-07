import torch

from dataset import ShapeNetRenderer, sample_from_clouds, mask_from_depth, add_gaussian_noise

from utils.visualization import viz_image_batch
from utils.load_save import save_model

from diffusers.optimization import get_cosine_schedule_with_warmup

from models.vae import VAEConvNet, vae_loss, VAEPointNetEncoder

from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
import pathlib as pl
import wandb
import hydra
import os

def train(config):

    # Logger
    if config.log: 
        wandb.init(**config.logger, config=OmegaConf.to_container(config))
        log = wandb.log
    else: log = lambda x: None
    
    # Setup Dataset
    render = ShapeNetRenderer.build(
                            path=         config.dataset.path,
                            categories=   config.dataset.categories,
                            split=        config.dataset.split,
                            num_objects=  config.dataset.num_objects,
                            batch_size=   config.dataset.num_views,
                            image_size=   config.dataset.image_size,
                            shuffle=      config.dataset.shuffle,
                            device=       config.dataset.device, 
                            scale_mode=   config.dataset.scale_mode,
                            cloud_post_process=add_gaussian_noise,
                            )
    
    channel_size = 1
    input_shape = (channel_size, config.dataset.image_size, config.dataset.image_size)
    model = VAEPointNetEncoder(input_shape, config.depth_latent_size).to(config.device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=(config.num_epochs),
    )
    epoch = 0
    batch_tqdm = tqdm(range(config.num_epochs), 
                      desc='Training Step Loop')
    for batch_i in batch_tqdm:
        log({'step': batch_i+1})

        renders = render()

        depths = renders['depths'].permute(0, 3, 1, 2).to(config.device)
        y = mask_from_depth(depths, inverse=True)
        x = sample_from_clouds(renders['face_points'], 1000).to(config.device)
        
        y_hat, z_mu, z_log_std = model(x)
        loss = vae_loss(y, y_hat, z_mu, z_log_std)

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

                
    as_np = lambda x: (x/2 + 0.5).permute(0,2,3,1).detach().cpu().numpy()
    # if x.shape[1] == 2: x, x_hat = x[:, 0:1], x_hat[:, 0:1]
    viz_image_batch(as_np(y), block=False, title='Original')
    viz_image_batch(as_np(y_hat), title='Reconstructed')
    pass


@hydra.main(version_base=None, config_path='./config', config_name='depth_vae')
def run(cfg: DictConfig) -> None:
    if cfg.log_locally:
        os.environ["WANDB_MODE"] = "offline"
    train(cfg)

if __name__ == '__main__':
    run()