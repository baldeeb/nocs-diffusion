import os
import json
import torch
import zipfile
from torch.utils.data import Dataset
from pytorch3d.io import load_objs_as_meshes

from ..synsetids import synsetid_to_cate, cate_to_synsetid
from .defaults import DEFAULT_SPLIT_PERCENTAGES

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
                 preload=False # Load all meshes to memory
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
        self.data, self.meta = self._load_data() if self.preload else None, None

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
                obj_ids = [dir.split('/')[-3] for dir in obj_files]

                n_samples = len(obj_files)
                idxs = torch.randperm(n_samples).numpy()
                
                train_count = int(n_samples * self.split_percentages['train'])
                val_count = int(n_samples * self.split_percentages['val'])
                
                split_meta['train'] = [(synset_id, obj_ids[i], obj_files[i]) 
                                       for i in idxs[:train_count]]
                split_meta['val']   = [(synset_id, obj_ids[i], obj_files[i]) 
                                       for i in idxs[train_count: train_count + val_count]]
                split_meta['test']  = [(synset_id, obj_ids[i], obj_files[i])
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
            ids = [dir.split('/')[-3] for dir in obj_files]
            data[synset_id], meta[synset_id] = self.get_data_and_meta_dicts(ids, obj_files)
            # objs = load_objs_as_meshes(obj_files, device=self.device, 
            #                         create_texture_atlas=True,
            #                         load_textures=True)
            # data[synset_id] = {id: obj for id, obj in zip(ids, objs)}
            # meta_files = [f"{f[:-4]}.json" for f in obj_files]
            # for meta_file in meta_files:
            #     with open(meta_file, 'r') as f:
            #         object_meta = json.load(f)
            #         meta[synset_id][object_meta['id']] = object_meta 

        return data, meta
    
    def get_data_and_meta_dicts(self, obj_ids, obj_files):
        data, meta = {}, {}
        objs = load_objs_as_meshes(obj_files, device=self.device, 
                                    create_texture_atlas=True,
                                    load_textures=True)
        data = {id: obj for id, obj in zip(obj_ids, objs)}

        meta_files = [f"{f[:-4]}.json" for f in obj_files]
        for meta_file in meta_files:
            with open(meta_file, 'r') as f:
                object_meta = json.load(f)
                meta[object_meta['id']] = object_meta 
                # object_meta['centroid'] = torch.tensor(object_meta['centroid'])
                # object_meta['max'] = torch.tensor(object_meta['max'])
                # object_meta['min'] = torch.tensor(object_meta['min'])
        return data, meta
    
    def __len__(self):
        return len(self.split_info[self.split])

    def __getitem__(self, idx):
        synset_id, file_id, file_path = self.split_info[self.split][idx]
        if self.preload:
            mesh = self.data[synset_id][file_id]
            meta_data = self.meta[synset_id][file_id]
            return mesh, synset_id, file_id, meta_data
        else:
            obj, meta_data = self.get_data_and_meta_dicts([file_id], [file_path])
            return obj[file_id], synset_id, file_id, meta_data[file_id]


