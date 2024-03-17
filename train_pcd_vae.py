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

from utils.train import train

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
    class CloudToMaskVae(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.net = model
        def forward(self, **data):
            return self.net(data['face_points'])
        def loss(self, **data):
            return self.net.loss(data['face_points'], data['masks'])
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=cfg.lr_warmup_steps,
        num_training_steps=(cfg.num_epochs),
    )

    dataloader = hydra.utils.instantiate(cfg.dataloader).to(cfg.device)

    train(cfg, CloudToMaskVae(model), optimizer, lr_scheduler, dataloader)

    visualize_sample(dataloader, model)

if __name__ == '__main__':
    run()