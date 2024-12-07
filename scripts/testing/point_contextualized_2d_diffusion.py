from torchview import draw_graph

from nocs_diffusion import ConfigDirectoriesManager, ConfigLoader

device = 'cuda'
GRAPH_OUT_FORMAT = 'png'
OUT_DIRECTORY = './data/sandbox/'

def test_and_graph_model():
    cfg_path = ConfigDirectoriesManager()['train_nocs_diffuser.yaml']
    loader = ConfigLoader.from_config_path(str(cfg_path))
    dataloader = loader.dataloader
    model = loader.model.to(device)
    data = dataloader()
    
    images = data['images']
    data['timesteps'] = model.sample_timesteps(len(images), images.device)
    data['noisy_images'] = model.add_noise(images, data['timesteps'])
    output = model(**data)
    print(f'result shape: {output.shape}')
    
    model_graph = draw_graph(model, input_data=(data), expand_nested=True, depth=9)
    model_graph.visual_graph.render('PointContextualized2dDiffusionModel', 
                                    directory=OUT_DIRECTORY,
                                    format=GRAPH_OUT_FORMAT)

if __name__ == "__main__":
    test_and_graph_model()