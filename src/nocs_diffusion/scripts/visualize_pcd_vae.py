
from nocs_diffusion.utils import viz_image_batch, ConfigDirectoriesManager, ConfigLoader
from argparse import ArgumentParser

def visualize_sample(dataloader, model):
    data = dataloader()
    y_hat, _, _, = model(**data)
    as_np = lambda x: (x/2 + 0.5).permute(0,2,3,1).detach().cpu().numpy()
    viz_image_batch(as_np(data['masks']), block=False, title='Original')
    viz_image_batch(as_np(y_hat), title='Reconstructed')


def run():
    parser = ArgumentParser(
                    prog='visualize_nocs_diffusion',
                    description='Visualizes the results of a trained nocs diffusion model.',
                    epilog='Text at the bottom of help')
    parser.add_argument("-c", "--checkpoint", 
                    help="First argument can be used to point to a checkpoint." + \
                         "The director of the checkpoint is expected to house" + \
                         ".hydra/config.yaml",)
    args = parser.parse_args()
    if args.checkpoint is not None:
        loader = ConfigLoader.load_from_checkpoint(args.checkpoint)
    else:
        cfg_path = ConfigDirectoriesManager()['eval_depth_vae.yaml']
        loader = ConfigLoader.from_config_path(str(cfg_path))
    model = loader.model
    dataloader = loader.dataloader
    visualize_sample(dataloader, model)

if __name__ == '__main__':
    run()