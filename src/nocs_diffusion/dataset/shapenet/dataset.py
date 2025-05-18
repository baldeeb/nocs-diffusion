import os
import json
import torch
import zipfile
from torch.utils.data import Dataset
from ..utils.object_handler import ObjHandler

from ..synsetids import synsetid_to_cate, cate_to_synsetid
from . import DEFAULT_SPLIT_PERCENTAGES

class ShapeNetDataset(Dataset):
    '''
    
    '''
    def __init__(self,
                 root_dir, # Path to the ShapeNet dataset
                 synset_ids=None, # List of synset IDs to load 
                 synset_names=None, # List of synset IDs to load 
                 split='train', # {'train', 'val', 'test'} 
                 split_percentages=DEFAULT_SPLIT_PERCENTAGES, # {Train: %, Val: %, Test: %}
                 device='cpu', # Allows user to laod all meshes to a device
                 verbose=False, # Print loading information
                 preload=False, # Load all meshes to memory
                 **kwargs # Other arguments
                ):
        self.preload = preload
        self.device = device
        self.root_dir = root_dir
        self.split = split
        self.split_percentages = split_percentages

        if synset_ids:
            self.synset_ids = synset_ids
        elif synset_names:
            self.synset_ids = [cate_to_synsetid[name] for name in synset_names]

        if verbose:
            print(f"Loading ShapeNet dataset with synset IDs: " + 
                  f"{[synsetid_to_cate[id] for id in self.synset_ids]}")
        
        self._unzip_objects_as_needed()
        self.split_info = self._load_split_meta() # {split: LIST[(synset_id, obj_id, file_location), ....]}
        (self.data, self.meta) = self._load_data() if self.preload else (None, None)

        if verbose:
            print(f"Loaded {len(self.split_info[self.split])} samples for split {self.split}")

    def _unzip_objects_as_needed(self):
        for synset_id in self.synset_ids:
            folder_path = os.path.join(self.root_dir, f"{synset_id}")            
            if not os.path.isdir(folder_path):
                zip_path = f"{folder_path}.zip"
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(self.root_dir)

    def _load_split_meta(self):
        '''Loads the split file for the given synset IDs, 
        if the user is requesting the same IDs previously seen. 
        Otherwise, creates a pt file with the split.'''
        requested_synsets = '_'.join(self.synset_ids)
        split_file = os.path.join(self.root_dir, f"meta_dataloader_{requested_synsets}.pt")
        if os.path.exists(split_file):
            split_meta = torch.load(split_file)
        else:
            split_meta = {'train':[], 'val':[], 'test':[]}
            
            for synset_id in self.synset_ids:
                folder_path = os.path.join(self.root_dir, f"{synset_id}")            
                obj_files = [f"{d}/{f}" for d, _, files in os.walk(folder_path) 
                                        for f in files 
                                        if f.endswith('.obj')]
                fids = [dir.split('/')[-3] for dir in obj_files]

                n_samples = len(obj_files)
                idxs = torch.randperm(n_samples).numpy()
                
                train_count = int(n_samples * self.split_percentages['train'])
                val_count = int(n_samples * self.split_percentages['val'])
                
                split_meta['train'] = [(synset_id, fids[i], obj_files[i]) 
                                       for i in idxs[:train_count]]
                split_meta['val']   = [(synset_id, fids[i], obj_files[i]) 
                                       for i in idxs[train_count: train_count + val_count]]
                split_meta['test']  = [(synset_id, fids[i], obj_files[i])
                                       for i in idxs[train_count + val_count:]]
                torch.save(split_meta, split_file)
        
        return split_meta

    
    def _load_data(self):
        data = {}
        meta = {}
        for synset_id in self.synset_ids:
            data[synset_id] = {}
            meta[synset_id] = {}
            folder_path = os.path.join(self.root_dir, f"{synset_id}")            
            obj_files = [f"{d}/{f}" for d, _, files in os.walk(folder_path) 
                                    for f in files 
                                    if f.endswith('.obj')]
            data[synset_id], meta[synset_id] = self.get_data_and_meta_dicts(obj_files)
        return data, meta
    
    def get_data_and_meta_dicts(self, obj_files):
        # TODO: Separate get data and get meta
        data, meta = {}, {}
        objs = []
        for f in obj_files:
            o = ObjHandler(f)
            o.colorize_nocs()
            objs.append(o)        
        
        # Read meta data files
        path_splits = [f.split('/') for f in obj_files]
        file_ids = [s[-3] if s[-2] == 'models' else s[-2] for s in path_splits]
        for f, fid, obj in zip(obj_files, file_ids, objs): 
            data[fid] = obj            
 
            shapenet_meta = os.path.exists(f"{f[:-4]}.json")
            nocs_meta = os.path.exists(f"{os.path.dirname(f)}/bbox.txt")
            if shapenet_meta:
                with open(f"{f[:-4]}.json", 'r') as f:
                    m = json.load(f)
                    # scale of the centered normalized object
                    m['scale'] = (torch.tensor(m['max']) -
                                  torch.tensor(m['min'])
                                 ).norm()
                    meta[fid] = m 
            elif nocs_meta:
                # NOTE: not yet set to handle this.
                # read f"{os.path.dirname(f)}/bbox.txt"
                meta[fid] = {}
            else:
                meta[fid] = {} 

        return data, meta
    
    def __len__(self):
        return len(self.split_info[self.split])

    def __getitem__(self, idx):
        synset_id, file_id, file_path = self.split_info[self.split][idx]
        if self.preload:
            obj = self.data[synset_id][file_id]
            meta_data = self.meta[synset_id][file_id]
        else:
            obj, meta_data = self.get_data_and_meta_dicts([file_path])
            obj, meta_data = obj[file_id], meta_data[file_id]

        return obj, synset_id, file_id, meta_data

