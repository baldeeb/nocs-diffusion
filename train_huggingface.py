import torch
import torch.nn.functional as F

from models.nocs_diffusion import NocsDiff, CtxtEncoder
from models.scheduler import VarianceSchedule
from models.unets import get_unet

from utils.renderer import RendererWrapper, sample_transforms, mask_from_depth
from utils.dataloader import PointCloudLoader
from utils.nocs_generator import nocs_extractor
from utils.visualization import viz_image_batch

from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers import DDPMScheduler

from torchvision import transforms

def train(config):

    def add_noise(x, mu=0, std=0.005):
        return x + torch.randn(x.shape, device=x.device) * std + mu

    # Setup Dataset
    dataloader = PointCloudLoader(
                            path='./data/shapenet.hdf5',
                            categories=['chair'],
                            split='test',
                            batch_size=config.num_objects,
                            shuffle=False,
                            device=config.device, 
                            post_process=add_noise
                        )

    # Setup Forward Diffusion
    scheduler = DDPMScheduler()
    ctx_net = CtxtEncoder(1, 64).to(config.device)
    
    
    
    
    # model = NocsDiff(3, ctx_net, 64).to(config.device)  # MY OWN
    model = get_unet(config.as_dict())
    model = model.to(config.device)

    render = RendererWrapper(image_size=config.image_size)

    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=(config.num_epochs),
    )
    scheduler.set_timesteps(config.num_train_timesteps)

    # Run Rendering
    pts = dataloader()
    feats = nocs_extractor(pts)

    # select a single object
    pts = pts[0]
    feats = (feats[0] - 0.5) * 2  # try range (-1, +1)
    
    for epoch in range(config.num_epochs):

        Rs, Ts = sample_transforms(config.num_views, elev_range=[0, 70])
        renders = render(pts, feats, Rs=Rs, Ts=Ts)
        images, depths = renders['images'], renders['depths']
        
        # # TEMP TEST: Setting all images to be the same. noise will be differen.
        # images[1:] = images[0:1]
        # depths[1:] = depths[0:1]
        
        images = images.permute(0, 3, 1, 2).to(config.device)
        depths = depths.permute(0, 3, 1, 2).to(config.device)
        masks = mask_from_depth(depths)

        # VISUALIZE DATA
        # viz_image_batch(images.permute(0,2,3,1).detach().cpu().numpy())


        # forward diffusion
        timesteps = torch.randint(
            0, config.num_train_timesteps, (images.shape[0],), device=config.device, dtype=torch.int64
        )
        noise = torch.randn_like(images)
        noised_images = scheduler.add_noise(images, noise, timesteps)
        # viz_image_batch(noised_images.permute(0,2,3,1).detach().cpu().numpy())
        

        # pred_noise = model(noised_images, depths)  # MY OWN
        pred_noise = model(noised_images, timesteps).sample  # Unconditioned UNET
        loss = F.mse_loss(noise, pred_noise)
        loss.backward()
        
        # print loss
        if epoch % 500 == 0: print(loss)
        
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()


    as_np = lambda x: (x/2 + 0.5).permute(0,2,3,1).detach().cpu().numpy()

    viz_image_batch(as_np(images), block=False, title='Original')
    viz_image_batch(as_np(noised_images), block=False, title='Noised')

    img = noised_images.clone().detach()
    for t in scheduler.timesteps:
        with torch.no_grad():
            # noisy_residual = model(input, depths)  # MY OWN
            noisy_residual = model(img, t).sample  # UNet
        previous_noisy_sample = scheduler.step(noisy_residual, t, img).prev_sample
        img = previous_noisy_sample
    
    viz_image_batch(as_np(img), title='Fixed')
    pass


@hydra.main(version_base=None, config_path='./config', config_name='diffuser')
def run(cfg: DictConfig) -> None:
    if cfg.log_locally:
        os.environ["WANDB_MODE"] = "offline"
    train(cfg)

if __name__ == '__main__':
    run()