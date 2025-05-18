from  nocs_diffusion.dataset import ShapeNetDataloader, ShapeNetRenderer, add_gaussian_noise, RandSe3NoiseFunctor
from nocs_diffusion.utils import viz_image_batch


if __name__=='__main__':
  renderer = ShapeNetRenderer.build(
    path = './data/shapenet.hdf5',
    categories = ['mug', 'bowl', 'laptop', 'bottle', 'camera'],
    category_ids = [0, 1, 2, 3, 4],
    split = 'test',
    num_objects = None,    # count of objects to use, None implies all
    batch_size = 18,     # count of poses from which to view object
    image_size = 32,
    shuffle = True,     # whether to shuffle when choosing objects
    scale_mode = None, 
    post_process = add_gaussian_noise,
    pt_radius = 0.3,
    pts_per_pxl = 5,
    # feature_extractor= lambda *args, **kwargs : 0 
  )

  se3noiseaug = RandSe3NoiseFunctor(
    dist_range=[-0.2, 0.2],
    elev_range=[0,    70 ],
    azim_range=[0,    360],
  )

  dataloader = ShapeNetDataloader(
    renderer = renderer,
    return_dict = ['face_points', 'masks', 'images'],
    augment_face_points = se3noiseaug,
    points_per_render = 1000
  )


  as_np = lambda x: (x/2 + 0.5).permute(0,2,3,1).detach().cpu().numpy()

  i = ''
  while i != 'q':
    d = dataloader()
    viz_image_batch(as_np(d['images']), title='NOCS Images')
    i = input("Type q to quit or anything else to produce another image.")