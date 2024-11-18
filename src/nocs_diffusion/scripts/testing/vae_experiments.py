
from nocs_diffusion.utils import viz_image_batch, ConfigDirectoriesManager, ConfigLoader
from argparse import ArgumentParser

def visualize_sample(dataloader, model):
    data = dataloader()
    y_hat, _, _, = model(**data)
    as_np = lambda x: (x/2 + 0.5).permute(0,2,3,1).detach().cpu().numpy()
    viz_image_batch(as_np(data['masks']), block=False, title='Original')
    viz_image_batch(as_np(y_hat), title='Reconstructed')


if __name__ == '__main__':

    cfg_path = ConfigDirectoriesManager()['eval_depth_vae.yaml']
    loader = ConfigLoader.from_config_path(str(cfg_path))
    model = loader.model
    dataloader = loader.dataloader
    visualize_sample(dataloader, model)
