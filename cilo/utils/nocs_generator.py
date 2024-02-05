from utils.nocs_renderer import NOCSRenderer
from utils.dataset import ShapeNetCore
from utils.data import DataLoader, get_data_iterator
import torch

class PCloudLoader:
    def __init__(self, 
                path,
                categories,
                batch_size=1,
                shuffle=False,
                split='test',
                num_workers = 0,
                scale_mode='centered_nocs',
                device='cuda',
            ):
            dataset = ShapeNetCore(
                path=path,
                cates=categories,
                split=split,
                scale_mode=scale_mode
            )
            dataloader = DataLoader(dataset, 
                                    batch_size=batch_size, 
                                    num_workers=num_workers, 
                                    shuffle=shuffle)
            self._loader = dataloader
            self._data_itr = get_data_iterator(dataloader)
            self._device = device
        
    @property
    def batch_size(self):
        return self._loader.batch_size

    def __call__(self):
        batch = next(self._data_itr)
        return batch['pointcloud'].to(self._device)

def shift_half_unit(x:torch.Tensor):
    return x + 0.5

def mask_extractor(x:torch.Tensor):
    return (x.abs().sum(dim=-1, keepdim=True) > 0).float()

class NOCSGenerator:
    def __init__(self, dataloader,
                renderer=NOCSRenderer(), 
                feature_extractor=None, 
                device='cuda'
                ):
        assert dataloader.batch_size == 1, "Generator can only handle batches of 1 currently..."
        self._loader = dataloader
        self._extractor = feature_extractor
        self._renderer = renderer
        self._device = device
    
    def _extract_features(self, x):
        if self._extractor is None:
            # s = list(x.size())
            # s[-1] = 0
            # return torch.tensor(s)
            return None
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


# TODO: Get rid of this now redundant thing
def get_data_generator(path,
                       cates,
                       split,
                       shuffle=False,
                       device='cuda',
                       batch_size=1):
    dataloader = PCloudLoader(
        path,
        cates,
        batch_size=batch_size,
        shuffle=shuffle,
        split=split,
        device=device
    )
    renderer = NOCSRenderer()
    return NOCSGenerator(dataloader, renderer)