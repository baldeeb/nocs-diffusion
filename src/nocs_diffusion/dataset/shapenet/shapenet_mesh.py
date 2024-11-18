

def load_obj(self):

    for synsetid in self.cate_synsetids:
        base_path = ""
        path = base_path + f"/{synsetid}.obj"
        mesh = IO().load_mesh(path)
        
        cate_name = synsetid_to_cate[synsetid]
        
        for j, pc in enumerate(f[synsetid][self.split]):
            yield torch.from_numpy(pc), j, cate_name
        
