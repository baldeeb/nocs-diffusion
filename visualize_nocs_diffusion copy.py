from utils.visualization import viz_image_batch
from omegaconf import DictConfig
import hydra


def visualize(dataloader, model):
    as_np = lambda x: (x/2 + 0.5).permute(0,2,3,1).detach().cpu().numpy()
    data = dataloader()
    
    images = data['images']
    viz_image_batch(as_np(images), block=False, title='Original')
    
    noised_images = model.add_noise(images)
    viz_image_batch(as_np(noised_images), block=False, title='Noised')
    
    fixed_images = model.fix(noisy_images=noised_images, **data)
    viz_image_batch(as_np(fixed_images), title='Fixed')


@hydra.main(version_base=None, config_path='./config', config_name='eval_diffusion')
def run(cfg: DictConfig) -> None:
    dataloader = hydra.utils.instantiate(cfg.dataloader).to(cfg.device)
    model = hydra.utils.instantiate(cfg.diffusion_model).to(cfg.device)
    visualize(dataloader, model)

if __name__ == '__main__':
    run()