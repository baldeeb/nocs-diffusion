import torch
import torch.nn.functional as F

from pytorch3d.ops import estimate_pointcloud_normals

from utils.nocs_renderer import RendererWrapper, sample_transforms, mask_from_depth
from utils.dataloader import PointCloudLoader
from utils.visualization import viz_image_batch

from utils.nocs_generator import nocs_extractor

from diffusers.optimization import get_cosine_schedule_with_warmup

from models.vae import VAEConvNet, vae_loss


def train():

    def add_noise(x, mu=0, std=0.005):
        return x + torch.randn(x.shape, device=x.device) * std + mu

    ######## Configs ############
    class Config:
        device = 'cuda'

        # Models
        depth_latent_size = 1028

        # Augmentation
        image_size = 32
        num_views = 128
        
        # Training
        num_objects = 2
        lr = 1e-3
        lr_warmup_steps = 50
        num_epochs = 10000
    config = Config()
    #############################

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
    channel_size = 2
    input_shape = (channel_size, config.image_size, config.image_size)
    model = VAEConvNet(input_shape, config.depth_latent_size)
    model = model.to(config.device)
    render = RendererWrapper(image_size=config.image_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=(config.num_epochs),
    )

    # Run Rendering
    pts = dataloader()
    feats = nocs_extractor(pts)
    normals = estimate_pointcloud_normals(pts).mean(0)

    # select a single object
    pts = pts[0]
    feats = (feats[0] - 0.5) * 2  # try range (-1, +1)
    
    for epoch in range(config.num_epochs):

        Rs, Ts = sample_transforms(config.num_views, elev_range=[0, 70])
        renders = render(pts, feats, normals=normals, Rs=Rs, Ts=Ts)
        
        x = renders['depths'].permute(0, 3, 1, 2).to(config.device)
        masks = mask_from_depth(x, inverse=True)
        x = torch.concatenate([x, masks], dim=1)
        
        # x = renders['depths'].permute(0, 3, 1, 2).to(config.device)
        # masks = mask_from_depth(x)
        # depth_mean = x.sum((1,2,3)) / masks.sum((1,2,3))
        # for i in range(len(x)): x[i][masks[i]==1] -= depth_mean[i]
        # x[masks==0] = 10.0

        # x = renders['normals'].permute(0, 3, 1, 2).to(config.device)
        # x = renders['images'].permute(0, 3, 1, 2).to(config.device)
        
        x_hat, z_mu, z_log_std = model(x)
        loss = vae_loss(x, x_hat, z_mu, z_log_std)

        # print loss
        if epoch % 100 == 0: print(loss)
        
        optimizer.zero_grad()
        loss['loss'].backward()
        optimizer.step()
        lr_scheduler.step()

    as_np = lambda x: (x/2 + 0.5).permute(0,2,3,1).detach().cpu().numpy()
    viz_image_batch(as_np(x), block=False, title='Original')
    viz_image_batch(as_np(x_hat), title='Reconstructed')

    pass
train()