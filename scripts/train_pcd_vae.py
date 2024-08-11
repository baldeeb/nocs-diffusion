import torch
from nocs_diffusion import train, ConfigDirectoriesManager, ConfigLoader
from diffusers.optimization import get_cosine_schedule_with_warmup

if __name__ == '__main__':
    cfg_path = ConfigDirectoriesManager()['train_depth_vae.yaml']
    loader = ConfigLoader.from_config_path(str(cfg_path))
    dataloader = loader.dataloader
    model = loader.model

    # TODO: move optimizer to hydra config.
    optimizer = torch.optim.AdamW(model.parameters(), lr=loader.cfg.lr)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=loader.cfg.lr_warmup_steps,
        num_training_steps=(loader.cfg.num_steps),
    )
    train(loader.cfg, model, optimizer, lr_scheduler, dataloader)
    