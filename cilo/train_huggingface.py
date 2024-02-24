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
    # var_sched = VarianceSchedule(beta_1, beta_max, steps)
    ctx_net = CtxtEncoder(1, 32).to(device)
    model = NocsDiff(3, ctx_net, 32).to(device)

    # Setup Renderer
    render = RendererWrapper()

    # Run Rendering
    pts = dataloader()
    feats = nocs_extractor(pts)
    
    # forward diffusion
    Rs, Ts = sample_transforms(1)
    renders = render(pts, feats, Rs=Rs, Ts=Ts)

    images = renders['images']
    images = scheduler.add_noise(images, torch.randn_like(images), torch.randint(0, 50, (len(images),)))
    # images = diffuser.fwd_diff(images)
    
    depths = renders['depths']
    masks = mask_from_depth(depths)
    images = images * masks

    viz_image_batch(images.detach().cpu().numpy())
    print("Visualization complete!")

    depths = depths.permute(0, 3, 1, 2).to(device)
    images = images.permute(0, 3, 1, 2).to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=1e-4)

    noise = torch.randn_like(images, device=images.device)
    for i in range(10000):
        # loss = diffuser.get_loss(images, depths)
        pred_noise = model(images, depths)
        loss = F.mse_loss(noise, pred_noise)
        if i % 500 == 0: print(loss)
        optim.zero_grad()
        loss.backward()
        optim.step()

    pass

    input = images.clone().detach()
    for t in scheduler.timesteps:
        with torch.no_grad():
            noisy_residual = model(input, depths)
        previous_noisy_sample = scheduler.step(noisy_residual, t, input).prev_sample
        input = previous_noisy_sample

train()