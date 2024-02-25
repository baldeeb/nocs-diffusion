import torch
import torch.nn.functional as F

from models.nocs_diffusion import NocsDiff, CtxtEncoder

from models.scheduler import VarianceSchedule
from utils.nocs_renderer import RendererWrapper, sample_transforms, mask_from_depth
from utils.dataloader import PointCloudLoader
from utils.nocs_generator import nocs_extractor
from utils.visualization import viz_image_batch

from diffusers import DDPMScheduler


def train():

    def add_noise(x, mu=0, std=0.005):
        return x + torch.randn(x.shape, device=x.device) * std + mu

    ######## Configs ############
    batch_size   = 4
    num_views = 4
    # Variance Scheduler 
    steps        = 100
    beta_1       = 1e-4
    beta_max     = 0.001
    device       = 'cuda'
    #############################

    # Stup Dataset
    dataloader = PointCloudLoader(
                            path='./data/shapenet.hdf5',
                            categories=['chair'],
                            split='test',
                            batch_size=batch_size,
                            shuffle=False,
                            device=device, 
                            post_process=add_noise
                        )

    # Setup Forward Diffusion
    scheduler = DDPMScheduler()
    ctx_net = CtxtEncoder(1, 32).to(device)
    model = NocsDiff(3, ctx_net, 32).to(device)
    render = RendererWrapper()
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4)

    scheduler.set_timesteps(50)

    # Run Rendering
    pts = dataloader()
    feats = nocs_extractor(pts)

    # select a single object
    pts = pts[0]
    feats = (feats[0] - 0.5) * 2  # try range (-1, +1)
    
    Rs, Ts = sample_transforms(num_views, elev_range=[0, 180])
    renders = render(pts, feats, Rs=Rs, Ts=Ts)
    images, depths = renders['images'], renders['depths']
    
    # TEMP TEST: Setting all images to be the same. noise will be differen.
    images[1:] = images[0:1]
    depths[1:] = depths[0:1]
    
    images = images.permute(0, 3, 1, 2).to(device)
    depths = depths.permute(0, 3, 1, 2).to(device)
    masks = mask_from_depth(depths)
    for epoch in range(2000):

        # forward diffusion
        noise = torch.randn_like(images, device=images.device)
        noised_images = scheduler.add_noise(images, noise, torch.randint(0, 50, (len(images),)))
        noised_images = noised_images * masks
        
        
        pred_noise = model(noised_images, depths)
        loss = F.mse_loss(noise, pred_noise)
        
        # print loss
        if epoch % 500 == 0: print(loss)
        
        optim.zero_grad()
        loss.backward()
        optim.step()

    input = noised_images.clone().detach()
    for t in scheduler.timesteps:
        with torch.no_grad():
            noisy_residual = model(input, depths)
        previous_noisy_sample = scheduler.step(noisy_residual, t, input).prev_sample
        input = previous_noisy_sample
    
    input = input/2 + 0.5
    viz_image_batch(input.permute(0,2,3,1).detach().cpu().numpy())
    pass
train()