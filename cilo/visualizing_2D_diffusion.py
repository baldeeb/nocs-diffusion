from utils.diffusion import (VarianceSchedule, Diffuser)

from utils.visualization import viz_image_batch
from PIL import Image

import numpy as np
import torch

# Add noise
steps = 100
var_sched = VarianceSchedule(1e-4, 0.25, steps)
diffuse = Diffuser(var_sched)

TEST_IMAGE = False
if TEST_IMAGE:
    image_path = './data/test_img.jpeg'
    image = Image.open(image_path)
    image = torch.from_numpy(np.array(image)) / 255.0
    image = image - 0.5
    print(f'image as tensor: {image.shape}')
    print(f'loading {image_path} -> shape: {image.size}')
    mask = None
else:
    from utils.nocs_generator import get_data_generator
    gen = get_data_generator(
        path='./data/shapenet.hdf5',
        cates=['chair'],
        split='test',
        num_renders=12,
    )
    data = gen()
    print(f'data type {type(data)}, shape {data.shape}')
    # viz_image_batch(data[:, :, :, 3:6])
    image = data[0, :, :, 3:6] - 0.5
    mask = torch.zeros_like(image)
    mask[image > -0.5] = 1
    mask = (mask[:, :, 0].bool() | mask[:, :, 1].bool() | mask[:, :, 2].bool())[:, :, None].float()  



diff_ims = torch.stack([diffuse(image, i) for i in range(0, steps, steps//12)])
diff_ims = diff_ims.clip(-0.5, 0.5) + 0.5

if mask is not None:
    diff_ims *= mask

viz_image_batch(diff_ims)