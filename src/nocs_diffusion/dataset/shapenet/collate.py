import torch 
from pytorch3d.structures import join_meshes_as_batch

class CollateFunctor:    
    def __init__(self, meshes=True, meta_keys=None):
        self.meshes = meshes
        self.meta_keys = meta_keys

    def __call__(self, batch):
        objects, synset_ids, file_ids, metas = zip(*batch)
        meta_batches = {}
        for k in self.meta_keys:
            meta_batches[k] = [m[k] for meta in metas 
                                   for m in meta.values()]
            meta_batches[k] = torch.stack(meta_batches[k])
        
        if self.meshes:
            objects = join_meshes_as_batch(objects)
        else: # assume clouds
            objects = torch.stack(objects)
            
        return objects, synset_ids, file_ids, meta_batches
