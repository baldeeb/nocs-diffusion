import torch

from models.diffusion import ConditionedDiffusionModel

from utils.train import train
from utils.visualization import viz_image_batch

from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers import DDPMScheduler

from omegaconf import DictConfig
import hydra
import os


def visualize(dataloader, model, ddpm_scheduler):
    as_np = lambda x: (x/2 + 0.5).permute(0,2,3,1).detach().cpu().numpy()
    data = dataloader()
    
    images = data['images']
    viz_image_batch(as_np(images), block=False, title='Original')
    
    noised_images = model.add_noise(images)
    viz_image_batch(as_np(noised_images), block=False, title='Noised')
    
    fixed_images = model.fix(noisy_images=noised_images, **data)
    viz_image_batch(as_np(fixed_images), title='Fixed')


@hydra.main(version_base=None, config_path='./config', config_name='diffuser')
def run(cfg: DictConfig) -> None:

    if cfg.log_locally:
        os.environ["WANDB_MODE"] = "offline"
    
    dataloader = hydra.utils.instantiate(cfg.dataloader).to(cfg.device)

    # Setup Forward Diffusion
    noise_sched = DDPMScheduler()
    noise_sched.set_timesteps(cfg.num_train_timesteps)
    
    # Load context encoder
    ctxt_net = hydra.utils.instantiate(cfg.vae).encoder
    diff_net = hydra.utils.instantiate(cfg.diffusion_model)
    model = ConditionedDiffusionModel(diff_net, ctxt_net, noise_sched).to(cfg.device)

    # Set up optimizer  TODO: move to hydra config.
    optimizer = torch.optim.AdamW(diff_net.parameters(), lr=cfg.lr)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=cfg.lr_warmup_steps,
        num_training_steps=(cfg.num_epochs),
    )

    # Run training
    train(cfg, model, optimizer, lr_scheduler, dataloader)
    visualize(dataloader, model, noise_sched)

if __name__ == '__main__':
    run()