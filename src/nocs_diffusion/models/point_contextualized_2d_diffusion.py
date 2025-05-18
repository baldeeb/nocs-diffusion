import torch
import torch.nn.functional as F

from diffusers import DDPMScheduler
from ..utils.alignment import simple_align
from ..utils.transforms import to_transform_3d
import wandb

class PointContextualized2dDiffusionModel(torch.nn.Module):
    '''This class packages the huggingface diffusion utils to facilitate use.'''
    def __init__(self, diffusion_model, context_model, scheduler:DDPMScheduler):
        super().__init__()
        self.ctxt_net = context_model
        self.diff_net = diffusion_model
        self.scheduler = scheduler
        self.num_ts = len(scheduler.betas)  # num_train_timesteps
    
    def fix(self, noisy_images, face_points, category_ids=None, 
            num_inference_steps=None, **kwargs):
        """Fixes the noisy images using the face points and, 
        optionally, category ids as context.
        Passes kwargs to the context model.
        """
        target = noisy_images.clone().detach()
        
        # update inference step count
        if num_inference_steps: 
            self.scheduler.set_timesteps(num_inference_steps)

        # get context encoding
        # TODO: decide if the category_ids should be passed to the
        #   diff net as context.
        ctxt = self.ctxt_net(face_points, ctxt=category_ids,
                            **kwargs)
        
        # reverse diffusion
        for t in self.scheduler.timesteps:
            with torch.no_grad():
                res = self.diff_net(target, t, ctxt).sample
            target = self.scheduler.step(res, t, target).prev_sample
        
        # reset inference step count
        if num_inference_steps: 
            self.scheduler.set_timesteps(self.num_ts)
        
        return target  # Noisy image after reverse diffusion 
    
    def sample_timesteps(self, count, device):
        return torch.randint(0, self.num_ts, (count,), 
                             device=device, dtype=torch.int64)

    def add_noise(self, images, timesteps=None):
        if timesteps is None:
            timesteps = self.sample_timesteps(len(images), images.device)
        noise = torch.randn_like(images)
        return self.scheduler.add_noise(images, noise, timesteps).clamp(-1.0, 1.0)

    def forward(self, **data):
        ctxt = self.ctxt_net(data['face_points'], 
                             ctxt=data.get('category_ids', None),
                             **data)
        return self.diff_net(data['noisy_images'], 
                             data['timesteps'],
                             ctxt).sample
        
    def loss(self, **data):
        # forward diffuse
        timesteps = torch.randint(0, self.num_ts, (data['images'].shape[0],), 
                                    device=data['images'].device, dtype=torch.int64)
        noise = torch.randn_like(data['images'])
        noised = self.scheduler.add_noise(data['images'], noise, timesteps)
        
        # prediction noise 
        pred = self.forward(noisy_images=noised, timesteps=timesteps, **data)

        # calculate loss
        loss = F.mse_loss(noise, pred)
        return {'loss':loss}

    def sample(self, num_inference_steps=None, **data):
        '''Samples an images from a noisy one. 
        Creates a noisy image if not provided.'''
        noisy = data.get('noisy', self.add_noise(data['images']))
        return self.fix(num_inference_steps=num_inference_steps, 
                        noisy_images=noisy, **data)

        
class PointContextualized2dDiffusionModelValidator:
    def __init__(self, dataloader, device='cuda', num_batches=1,
                 num_inference_steps=25):
        self.dataloader = dataloader.to(device)
        self.num_batches = num_batches
        self.num_inference_steps = num_inference_steps 
    
    def _as_np(self, x): 
        return (x/2 + 0.5).permute(0,2,3,1).detach().cpu().numpy()

    def __call__(self, model, log):
        for _ in range(self.num_batches):
            d = self.dataloader()
            
            # Get the noisy images
            d['noisy'] = model.add_noise(d['images'])
            
            # Sample fix
            pred = model.sample(**d, num_inference_steps = self.num_inference_steps) # (B, C, H, W)
            
            if 'perspective_2d_indices' in d:
                img2pcd_idxs = d['perspective_2d_indices']
                pred_pts = torch.stack([p[:,  i[:, 0], i[:, 1]].T 
                                        for p, i in zip(pred, img2pcd_idxs)])
                Rt, mean_dists, dists, scale = simple_align(d['face_points'], pred_pts)
                Rts = to_transform_3d(Rt, scale)
                if 'transforms' in d:
                    # if Rts.shape[-2] == 3: 
                    #     last_row = torch.ones(*Rts.shape[:-2], 1, 4) * torch.tensor([0, 0, 0, 1])
                    #     Rts = torch.concatenate((Rts, last_row.to(Rts.device)), dim=-2)
                    
                    I = torch.eye(4).to(Rts.device)
                    gt_Rts = d['transforms'].get_matrix().to(Rts.device)
                    residual_Rts = (I - ( Rts.get_matrix() @ gt_Rts ) ).norm()
                    
            
            log({"point cloud": [wandb.Object3D(v.clone().detach().cpu().numpy()) 
                                 for v in d['face_points']],
                 "ground_truth_nocs": wandb.Image(d['images']),
                 "noisy_nocs": wandb.Image(d['noisy']),
                 "predicted_nocs": wandb.Image(pred),
                 "validation": { 
                     'abs_mean_cloud_dists': mean_dists.mean(), # Good indication of the quality of the nocs map
                     'norm_residual_transform': residual_Rts    # how good is the map at providing transform.
                    }
                 })
     