import torch
from .vae import VAE
import wandb


class CloudToMaskVae(torch.nn.Module):
        def __init__(self, model:VAE):
            super().__init__()
            self.net = model

        @property
        def encoder(self): return self.net.encoder
        
        @property
        def decoder(self): return self.net.decoder

        def forward(self, **data):
            return self.net(data['face_points'])
        
        def loss(self, **data):
            return self.net.loss(data['face_points'], data['masks'])
        
class CloudToMaskVaeValidator:
    def __init__(self, dataloader, device='cuda', num_eval_batches=1):
        self.dataloader = dataloader.to(device)
        self.num_eval_batches = num_eval_batches
    
    def _as_np(self, x): 
        return (x/2 + 0.5).permute(0,2,3,1).detach().cpu().numpy()

    def __call__(self, model, log):
        for _ in range(self.num_eval_batches):
            d = self.dataloader()
            y_hat, _, _, = model(**d)
            log({"ground_truth": wandb.Image(d['masks']),
                 "predicted": wandb.Image(y_hat),
                 "validation loss": model.loss(**d)})
     