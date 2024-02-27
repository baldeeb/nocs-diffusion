import torch
import torch.nn.functional as F

from models.nocs_diffusion import NocsDiff, CtxtEncoder

from models.scheduler import VarianceSchedule
from utils.nocs_renderer import RendererWrapper, sample_transforms, mask_from_depth
from utils.dataloader import PointCloudLoader
from utils.nocs_generator import nocs_extractor
from utils.visualization import viz_image_batch

from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers import DDPMScheduler

from torchvision import transforms

def train():

    def add_noise(x, mu=0, std=0.005):
        return x + torch.randn(x.shape, device=x.device) * std + mu

    ######## Configs ############
    class Config:
        device       = 'cuda'

        # Augmentation
        image_size=32
        num_views = 4
        
        # Training
        batch_size   = 2
        lr = 1e-4
        lr_warmup_steps = 50
        num_epochs = 10000
        num_train_timesteps = 1000
    config = Config()
    #############################

    # Stup Dataset
    dataloader = PointCloudLoader(
                            path='./data/shapenet.hdf5',
                            categories=['chair'],
                            split='test',
                            batch_size=config.batch_size,
                            shuffle=False,
                            device=config.device, 
                            post_process=add_noise
                        )

    # Setup Forward Diffusion
    scheduler = DDPMScheduler()
    ctx_net = CtxtEncoder(1, 64).to(config.device)
    
    
    
    
    # model = NocsDiff(3, ctx_net, 64).to(config.device)  # MY OWN

    from diffusers import UNet2DModel
    model = UNet2DModel(
        sample_size=config.image_size,  # the target image resolution
        in_channels=3,  # the number of input channels, 3 for RGB images
        out_channels=3,  # the number of output channels
        layers_per_block=1,  # how many ResNet layers to use per UNet block
        # block_out_channels=(128, 128, 256, 256, 512, 512),  # the number of output channels for each UNet block
        block_out_channels=(128, 128),  # the number of output channels for each UNet block
        down_block_types=(
            "DownBlock2D",  # a regular ResNet downsampling block
            # "DownBlock2D",
            # "DownBlock2D",
            # "DownBlock2D",
            # "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",  # a regular ResNet upsampling block
            # "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
            # "UpBlock2D",
            # "UpBlock2D",
            # "UpBlock2D",
            "UpBlock2D",
        ),
    )
    model = model.to(config.device)

    render = RendererWrapper(image_size=config.image_size)

    
    optim = torch.optim.AdamW(model.parameters(), lr=config.lr)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optim,
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
    
    Rs, Ts = sample_transforms(config.num_views, elev_range=[0, 70])
    renders = render(pts, feats, Rs=Rs, Ts=Ts)
    images, depths = renders['images'], renders['depths']
    
    # TEMP TEST: Setting all images to be the same. noise will be differen.
    images[1:] = images[0:1]
    depths[1:] = depths[0:1]
    
    images = images.permute(0, 3, 1, 2).to(config.device)
    depths = depths.permute(0, 3, 1, 2).to(config.device)
    masks = mask_from_depth(depths)

    # VISUALIZE DATA
    # viz_image_batch(images.permute(0,2,3,1).detach().cpu().numpy())

    for epoch in range(config.num_epochs):

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
        
        optim.step()
        lr_scheduler.step()
        optim.zero_grad()

    img = noised_images.clone().detach()
    viz_image_batch(img.permute(0,2,3,1).detach().cpu().numpy())

    for t in scheduler.timesteps:
        with torch.no_grad():
            # noisy_residual = model(input, depths)  # MY OWN
            noisy_residual = model(img, t).sample  # Unet
        previous_noisy_sample = scheduler.step(noisy_residual, t, img).prev_sample
        img = previous_noisy_sample
    
    img = img/2 + 0.5
    viz_image_batch(img.permute(0,2,3,1).detach().cpu().numpy())
    pass
train()