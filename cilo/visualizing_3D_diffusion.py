'''
Say we have a depth image and ground truth object shape
can we retrieve rotation, translation, and scale of that
object from the observed points
'''

from utils.dataset import ShapeNetCore
from utils.misc import *
from utils.data import DataLoader, get_data_iterator

import matplotlib.pyplot as plt

from utils.diffusion import (VarianceSchedule, Diffuser)
from utils.temp_config import Config
from utils.nocs_renderer import NOCSRenderer
from utils.visualization import viz_image_batch


from utils.nocs_generator import get_data_generator
gen = get_data_generator(
        path='./data/shapenet.hdf5',
        cates=['chair'],
        split='test',
        num_renders=12,
    )
data = gen()
print(f'data type {type(data)}, shape {data.shape}')
viz_image_batch(data[:, :, :, 3:6])


conf = Config()
# Datasets and loaders
test_dset = ShapeNetCore(
    path='./data/shapenet.hdf5',
    cates=['chair'],
    split='test',
    scale_mode='shape_unit'
)

data_loader = DataLoader(test_dset, 
                         batch_size=conf.batch_size, 
                         num_workers=0, 
                         shuffle=False)
test_loader = get_data_iterator(data_loader)

print(f'Loading data...')
batch = next(test_loader)
ref = batch['pointcloud'].to(conf.device)
ref_min = ref.min(dim=1, keepdims=True).values
ref_max = ref.max(dim=1, keepdims=True).values
scale = (ref_max - ref_min).norm(dim=2)[..., None]
shift = ref_min
pts = (ref - shift) / scale
print(f'point shapes {pts.shape}')

# Variance Scheduler
num_steps   = 9
beta_1      = 1e-4
beta_max    = 0.05

# Add noise
var_sched = VarianceSchedule(beta_1, beta_max, num_steps)
diffuse = Diffuser(var_sched)

if False:
    fig = plt.figure()
    for i in range(num_steps+1):
        pts_bar = diffuse(pts, i)
        axs = fig.add_subplot(1, num_steps+1, i+1, projection='3d')
        np_pts_bar = pts_bar.clone().detach().cpu().numpy()
        axs.scatter(np_pts_bar[0, :, 0], np_pts_bar[0, :, 1], np_pts_bar[0, :, 2])
    plt.show()
else:
    pts_bar = diffuse(pts, num_steps-1)

cols, rows = 3, 3
render_count = cols * rows
render = NOCSRenderer(render_count)

print('\n', '*'*25, '\n', 
    f'Rendering {render_count} random angles.')
images = render(pts[0], pts_bar[0])
fig, axs = plt.subplots(cols, rows)
for i, img in enumerate(images):
    r, c = i // rows, i % cols
    img = img.detach().cpu().numpy()
    axs[r][c].imshow(img[:, :, :3])
plt.show()


print('\n', '*'*25,  '\n', 
    f'Rendering different diffusion stages.')
rows = torch.sqrt(torch.tensor([num_steps]))
cols, rows = rows.ceil().int().item(), rows.floor().int().item()
fig, axs = plt.subplots(rows, cols)
for i in range(num_steps):
    r, c = i // rows, i % cols
    pts_bar = diffuse(pts, i)
    images = render(pts[0], pts_bar[0])
    images = images.detach().cpu().numpy()
    axs[r][c].imshow(images[0, :, :, 3:])
plt.show()
