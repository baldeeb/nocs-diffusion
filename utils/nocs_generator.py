from utils.renderer import RendererWrapper
import torch

def shift_half_unit(x:torch.Tensor):
    return x + 0.5

def mask_extractor(x:torch.Tensor):
    return (x.abs().sum(dim=-1, keepdim=True) > 0).float()

def nocs_extractor(x:torch.Tensor):
    '''
    Args:
        x: [B, W, H, 3]
    '''
    xmin = x.view(-1, x.shape[-1]).min(0).values
    xmax = x.view(-1, x.shape[-1]).max(0).values
    return (x / (xmax - xmin).norm()) + 0.5


def add_noise(x, mu, std):
    return torch.normal(mu, std, size=x.shape) + x
    
class NOCSObjectRenderer:
    def __init__(self, dataloader,
                renderer=RendererWrapper(), 
                feature_extractor=nocs_extractor, 
                device='cuda'
                ):
        assert dataloader.batch_size == 1, "Generator can only handle batches of 1 currently..."
        self._loader = dataloader
        self._extractor = feature_extractor
        self._renderer = renderer
        self._device = device
    
    def _extract_features(self, x):
        if self._extractor is None:
            return x
        else:
            return self._extractor(x)

    def __call__(self, num_renders=1):
        x = self._loader()
        # TODO: Fix batching
        #   - Decide if renderer needs to be able to take in batches.
        #   - If bathes should not be allowed the property should be enforced here.
        x = x[0]
        fx = self._extract_features(x)
        return self._renderer(x, fx, num_renders)