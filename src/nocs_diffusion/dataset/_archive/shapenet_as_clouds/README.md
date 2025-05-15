This dataloader is built to utilize data from {TODO GET LINK} which publishes a compacted version of a subset of shapenet that primary contains pointclouds.

Dataloader config might look like:

```
  _target_: nocs_diffusion.dataset.ShapeNetDataloader
  renderer:
    _target_: nocs_diffusion.dataset.ShapeNetRenderer.build
    path: './data/shapenet.hdf5'
    categories: ['mug', 'bowl', 'laptop', 'bottle', 'camera']
    category_ids: [0, 1, 2, 3, 4]
    split: test
    num_objects:    # count of objects to use, None implies all
    batch_size: 18     # count of poses from which to view object
    image_size: 32
    shuffle: true     # whether to shuffle when choosing objects
    scale_mode: 
    post_process:
      _target_: hydra.utils.get_method 
      path: nocs_diffusion.dataset.add_gaussian_noise
  return_dict:
    - face_points
    - masks
    - images
    - category_ids
    - perspective_2d_indices
    - transforms
  # augment_face_points:
  #   _target_: nocs_diffusion.dataset.RandSe3NoiseFunctor
  #   dist_range: [-0.2, 0.2]
  #   elev_range: [0,    70 ]
  #   azim_range: [0,    360]
  points_per_render: 1000
```