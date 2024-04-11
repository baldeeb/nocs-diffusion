from nocs_diffusion.utils import viz_image_batch, ConfigDirectoriesManager, ConfigLoader

def visualize(dataloader, model):
    as_np = lambda x: (x/2 + 0.5).permute(0,2,3,1).detach().cpu().numpy()
    data = dataloader()
    
    images = data['images']
    viz_image_batch(as_np(images), block=False, title='Original')
    
    noised_images = model.add_noise(images)
    viz_image_batch(as_np(noised_images), block=False, title='Noised')
    
    fixed_images = model.fix(noisy_images=noised_images, **data)
    viz_image_batch(as_np(fixed_images), title='Fixed')

if __name__ == '__main__':
    cfg_path = ConfigDirectoriesManager()['eval_diffusion.yaml']
    loader = ConfigLoader.from_config_path(str(cfg_path))

    dataloader = loader.dataloader
    model = loader.model
    visualize(dataloader, model)
