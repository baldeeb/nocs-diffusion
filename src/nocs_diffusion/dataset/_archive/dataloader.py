from ..shapenet.shapenet import ShapeNetCore
from ..data import DataLoader, get_data_iterator


class PointCloudLoader:
    def __init__(self, 
                path,
                categories,
                batch_size=1,
                shuffle=False,
                split='test',
                num_workers = 0,
                scale_mode=None, # centered_nocs, ...
                device='cuda',
                post_process=None,
            ):
            dataset = ShapeNetCore(path=path,
                                   cates=categories,
                                   split=split,
                                   scale_mode=scale_mode)
            self._loader = DataLoader(dataset, 
                                    batch_size=batch_size, 
                                    num_workers=num_workers, 
                                    shuffle=shuffle)
            self._data_itr = get_data_iterator(self._loader)
            self._post_process = post_process if post_process else lambda x:x
            self._device = device
        
    @property
    def batch_size(self):
        return self._loader.batch_size

    def __call__(self):
        batch = next(self._data_itr)['pointcloud']
        batch = batch.to(self._device)
        batch = self._post_process(batch)
        return batch
