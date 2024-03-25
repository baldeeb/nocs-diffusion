from utils.visualization import viz_image_batch
from omegaconf import DictConfig
import hydra


def visualize_sample(dataloader, model):
    data = dataloader()
    x, y = data['face_points'], data['masks']
    y_hat, _, _, = model(x)
    as_np = lambda x: (x/2 + 0.5).permute(0,2,3,1).detach().cpu().numpy()
    viz_image_batch(as_np(y), block=False, title='Original')
    viz_image_batch(as_np(y_hat), title='Reconstructed')


@hydra.main(version_base=None, config_path='./config', config_name='depth_vae')
def run(cfg: DictConfig) -> None:

    model = hydra.utils.instantiate(cfg.vae).to(cfg.device)
    dataloader = hydra.utils.instantiate(cfg.dataloader).to(cfg.device)

    visualize_sample(dataloader, model)

if __name__ == '__main__':
    run()