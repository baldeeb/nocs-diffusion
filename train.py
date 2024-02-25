import torch
import torch.nn.functional as F

from models.nocs_diffusion import NocsDiff, CtxtEncoder

from models.scheduler import VarianceSchedule
from utils.nocs_renderer import RendererWrapper, sample_transforms, mask_from_depth
from utils.dataloader import PointCloudLoader
from utils.nocs_generator import nocs_extractor
from utils.visualization import viz_image_batch




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
    var_sched = VarianceSchedule(beta_1, beta_max, steps)
    ctx_net = CtxtEncoder(1, 32).to(device)
    diffuser = NocsDiff(3, ctx_net, 32, var_sched).to(device)

    # Setup Renderer
    render = RendererWrapper()

    # Run Rendering
    pts = dataloader()
    feats = nocs_extractor(pts)

    # forward diffusion
    Rs, Ts = sample_transforms(1)
    renders = render(pts, feats, Rs=Rs, Ts=Ts)

    images = renders['images']
    images = diffuser.fwd_diff(images)
    
    depths = renders['depths']
    masks = mask_from_depth(depths)
    images = images * masks

    viz_image_batch(images.detach().cpu().numpy())
    print("Visualization complete!")

    depths = depths.permute(0, 3, 1, 2).to(device)
    images = images.permute(0, 3, 1, 2).to(device)

    optim = torch.optim.AdamW(diffuser.parameters(), lr=1e-4)

    for i in range(10000):
        loss = diffuser.get_loss(images, depths)
        if i % 500 == 0: print(loss)
        optim.zero_grad()
        loss.backward()
        optim.step()

    pass

train()