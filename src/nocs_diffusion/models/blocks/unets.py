import torch
from diffusers import UNet2DModel, UNet2DConditionModel

UNET_DEFAULTS = {
    'block_out_channels': (128, 128), # the number of output channels for each UNet block
    'down_block_types': ("CrossAttnDownBlock2D", "CrossAttnDownBlock2D"), # regular ResNet downsampling block
    'up_block_types': ("CrossAttnUpBlock2D", "CrossAttnUpBlock2D"), # regular ResNet upsampling block
    'cross_attention_dim': 512,
}
CONDITIONED_UNET_DEFAULTS = {
    'block_out_channels': (128, 128), # the number of output channels for each UNet block
    'down_block_types': ("CrossAttnDownBlock2D", "CrossAttnDownBlock2D"), # regular ResNet downsampling block
    'up_block_types': ("CrossAttnUpBlock2D", "CrossAttnUpBlock2D"), # regular ResNet upsampling block
    'cross_attention_dim': 512,
}

def get_unet(cfg):
    for k in UNET_DEFAULTS:
        if k not in cfg:
            cfg[k] = UNET_DEFAULTS[k]

    net = UNet2DModel(
        sample_size=cfg['image_size'],  # the target image resolution
        in_channels=3,  # the number of input channels, 3 for RGB images
        out_channels=3,  # the number of output channels
        layers_per_block=1,  # how many ResNet layers to use per UNet block
        block_out_channels=cfg['block_out_channels'],  
        down_block_types=cfg['down_block_types'],
        up_block_types=cfg['up_block_types']
    )
    if 'load_path' in cfg:
        sd = torch.load(cfg['load_path'])
        net.load_state_dict(sd)

def get_conditioned_unet(cfg):
    for k in CONDITIONED_UNET_DEFAULTS:
        if k not in cfg:
            cfg[k] = CONDITIONED_UNET_DEFAULTS[k]

    net = UNet2DConditionModel(
        sample_size=cfg['image_size'],
        in_channels=3,
        out_channels=3,
        layers_per_block=1,
        block_out_channels=cfg['block_out_channels'],  
        down_block_types=cfg['down_block_types'],
        up_block_types=cfg['up_block_types'],
        cross_attention_dim=cfg['cross_attention_dim']
    )
    if 'load_path' in cfg:
        sd = torch.load(cfg['load_path'])
        net.load_state_dict(sd)


if __name__ == '__main__':
    device = torch.device('cuda:0')
    net = get_conditioned_unet({'image_size':32}).to(device=device)
    images = torch.rand(2, 3, 32, 32, device=device)
    timesteps = torch.tensor([100, 100], dtype=torch.long, device=device)
    ctxts = torch.rand(2, 1, 1280, device=device)
    data = (images, timesteps, ctxts)
    result = net(*data).sample

    kwargs = {'input_names':'data', 'output_names':'results'}

    torch.onnx.export(net, 
                      (images, timesteps, ctxts), 
                      '.scratch/rnn.onnx', 
                      input_names=['data', 'timesteps', 'context'], 
                      output_names=['results'],
                    )
    # torch.onnx.export(net, *data, 'rnn.onnx', **kwargs)
