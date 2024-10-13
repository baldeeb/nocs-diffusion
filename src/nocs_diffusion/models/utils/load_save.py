import torch 
import os.path

def load_model(model, path):
    if path is not None:
        if not os.path.isfile(path):
            raise ValueError(f"{path} does not exist!")
        sd = torch.load(path)
        model.load_state_dict(sd)
    return model