import torch 

def load_model(model, path):
    if path is not None:
        sd = torch.load(path)
        model.load_state_dict(sd)
    return model