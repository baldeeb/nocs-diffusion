# ### An effort to nicify the diffuser training. 
# # temporarily abandoned for proof of concept.

# import torch

# from dataset import ShapeNetRenderer, sample_from_clouds, mask_from_depth, add_gaussian_noise

# from utils.visualization import viz_image_batch
# from utils.load_save import save_model

# from diffusers.optimization import get_cosine_schedule_with_warmup

# # from models.vae import VAEConvNet, vae_loss, VAEPointNetEncoder
# from models.nocs_diffusion import get_unet 

# from omegaconf import DictConfig, OmegaConf
# from tqdm import tqdm
# import pathlib as pl
# import wandb
# import hydra
# import os

# from utils.common import train



# @hydra.main(version_base=None, config_path='./config', config_name='depth_vae')
# def run(cfg: DictConfig) -> None:
#     if cfg.log_locally:
#         os.environ["WANDB_MODE"] = "offline"


#     # Setup Forward Diffusion
#     scheduler = DDPMScheduler()
#     scheduler.set_timesteps(cfg.num_train_timesteps)

#     ctx_net = CtxtEncoder(1, 64).to(cfg.device)
#     # model = NocsDiff(3, ctx_net, 64).to(config.device)  # MY OWN
#     model = get_unet(OmegaConf.to_object(cfg))
#     model = model.to(cfg.device)

#     optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
#     lr_scheduler = get_cosine_schedule_with_warmup(
#         optimizer=optimizer,
#         num_warmup_steps=cfg.lr_warmup_steps,
#         num_training_steps=(cfg.num_epochs),
#     )

#     # Setup Dataset
#     render = ShapeNetRenderer.build(
#                             path=         cfg.dataset.path,
#                             categories=   cfg.dataset.categories,
#                             split=        cfg.dataset.split,
#                             num_objects=  cfg.dataset.num_objects,
#                             batch_size=   cfg.dataset.num_views,
#                             image_size=   cfg.dataset.image_size,
#                             shuffle=      cfg.dataset.shuffle,
#                             device=       cfg.dataset.device, 
#                             scale_mode=   cfg.dataset.scale_mode,
#                             cloud_post_process=add_gaussian_noise,
#                             )

#     class Dataloader:
#         def __init__(self, renderer, cfg):
#             self.renderer = renderer
#             self.cfg = cfg
#         def __call__(self):
#             renders = self.renderer()
#             images = renders['images'].permute(0, 3, 1, 2).to(self.cfg.device)
        
            
#             # TODO: MOVE TO MODEL #################################################
#             # forward diffusion
#             timesteps = torch.randint(
#                 0, cfg.num_train_timesteps, (images.shape[0],), device=cfg.device, dtype=torch.int64
#             )
#             noise = torch.randn_like(images)
#             noised_images = scheduler.add_noise(images, noise, timesteps)
#             # viz_image_batch(noised_images.permute(0,2,3,1).detach().cpu().numpy())
#             ########################################################################

#             y = mask_from_depth(depths, inverse=True)
#             x = sample_from_clouds(renders['face_points'], 1000).to(self.cfg.device)
#             return x, y
#     dataloader = Dataloader(render, cfg)

#     train(cfg, model, optimizer, lr_scheduler, dataloader)

# if __name__ == '__main__':
#     run()