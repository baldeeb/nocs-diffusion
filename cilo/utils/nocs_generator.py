from utils.nocs_renderer import NOCSRenderer
from utils.dataset import ShapeNetCore
from utils.data import DataLoader, get_data_iterator



class NOCSGenerator:
    def __init__(self, dataloader, renderer, feature_extractor=None, device='cpu'):
        assert dataloader.batch_size == 1, "Generator can only handle batches of 1 currently..."
        self._data_itr = get_data_iterator(dataloader)
        self._extractor = feature_extractor
        self._renderer = renderer
        self._device = device
    
    def _get_pts(self):
        batch = next(self._data_itr)
        ref = batch['pointcloud'].to(self._device)
        ref_min = ref.min(dim=1, keepdims=True).values
        ref_max = ref.max(dim=1, keepdims=True).values
        scale = (ref_max - ref_min).norm(dim=2)[..., None]
        shift = ref_min
        return (ref - shift) / scale

    def _extract_features(self, x):
        if self._extractor is None:
            return x
        else:
            return self._extractor(x)

    def __call__(self):
        x = self._get_pts()

        # TODO: Fix batching
        #   - Decide if renderer needs to be able to take in batches.
        #   - If bathes should not be allowed the property should be enforced here.
        x = x[0]

        fx = self._extract_features(x)
        imgs = self._renderer(x, fx)
        return imgs


def get_data_generator(path,
                       cates,
                       split,
                       num_renders,
                       scale_mode='shape_unit',
                       shuffle=False
                       ):
    dataset = ShapeNetCore(
                path, 
                cates, 
                split, 
                scale_mode
            )
    dataloader = DataLoader(
                dataset, 
                batch_size=1, 
                num_workers=0, 
                shuffle=shuffle
            )
    renderer = NOCSRenderer(num_renders)
    return NOCSGenerator(dataloader, renderer)