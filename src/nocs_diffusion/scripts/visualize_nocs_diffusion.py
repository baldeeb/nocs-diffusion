from nocs_diffusion import viz_image_batch, ConfigDirectoriesManager, ConfigLoader
from omegaconf import OmegaConf
import wandb
import os

from argparse import ArgumentParser

def visualize(dataloader, model):
    as_np = lambda x: (x/2 + 0.5).permute(0,2,3,1).detach().cpu().numpy()
    data = dataloader()
    
    images = data['images']
    viz_image_batch(as_np(images), block=False, title='Original')
    
    noised_images = model.add_noise(images)
    viz_image_batch(as_np(noised_images), block=False, title='Noised')
    
    fixed_images = model.fix(noisy_images=noised_images, **data)
    viz_image_batch(as_np(fixed_images), title='Fixed')

def log_visuals(dataloader, model, log):
    data = dataloader()
    images = data['images']
    noised_images = model.add_noise(images)
    fixed_images = model.fix(noisy_images=noised_images, **data)

    results = {
        'Original': wandb.Image(images),
        'Noised':  wandb.Image(noised_images),
        'Fixed':  wandb.Image(fixed_images)
    }
    log(results)

def run():
    parser = ArgumentParser(
                    prog='visualize_nocs_diffusion',
                    description='Visualizes the results of a trained nocs diffusion model.',
                    epilog='Text at the bottom of help')
    parser.add_argument("-c", "--checkpoint", 
                    help="First argument can be used to point to a checkpoint." + \
                         "The director of the checkpoint is expected to house" + \
                         ".hydra/config.yaml",)
    args = parser.parse_args()
    if args.checkpoint is not None:
        loader = ConfigLoader.load_from_checkpoint(args.checkpoint)
    else:
        cfg_path = ConfigDirectoriesManager()['eval_diffusion.yaml']
        loader = ConfigLoader.from_config_path(str(cfg_path))
    cfg = loader.cfg
    dataloader = loader.dataloader
    model = loader.model

    if cfg.log:
        wandb.init(**cfg.logger, config=OmegaConf.to_container(cfg))
        log = wandb.log
        if cfg.log_locally: os.environ["WANDB_MODE"] = "offline"
        log_visuals(dataloader, model, log)
    else:
        visualize(dataloader, model)

if __name__ == '__main__':
    run()