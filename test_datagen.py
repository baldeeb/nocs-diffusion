from utils.dataset import *
from utils.misc import *
from utils.data import *

device = 'cpu'

# Datasets and loaders
test_dset = ShapeNetCore(
    path='./data/shapenet.hdf5',
    cates=['chair'],
    split='test',
    scale_mode='shape_bbox'
)
test_loader = get_data_iterator(DataLoader(test_dset, batch_size=1, num_workers=0))

batch = next(test_loader)
ref = batch['pointcloud'].to(device)
shift = batch['shift'].to(device)
scale = batch['scale'].to(device)

# Visual 3D point cloud
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_point_cloud(points, colors=None, show=True):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors)
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    if show: plt.show()

visualize_point_cloud(ref[0].detach().cpu().numpy())

# Generate a random SE(3) transformation
import numpy as np
from scipy.spatial.transform import Rotation

def random_SE3():
    # Random rotation
    r = Rotation.random()
    R = r.as_matrix()
    # Random translation
    t = np.random.rand(3) * 2 - 1
    t[-1] = 10
    # Random SE(3) transformation
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    # Scale the transformation
    T[:3, :3] *= 1.25
    return T


# Transform the point cloud
T = random_SE3()
ref = torch.bmm(ref, torch.from_numpy(T[:3, :3]).float().unsqueeze(0).to(device)) + torch.from_numpy(T[:3, 3]).float().unsqueeze(0).to(device)
points3d = ref[0].detach().cpu().numpy()

# Visualize the transformed point cloud
visualize_point_cloud(points3d)

# Define camera parameters
K = np.array([[552.41140, 0, 320.0],
                [0, 552.57043, 240.0],
                [0, 0, 1]])

# Project the point cloud to the image plane
def project_point_cloud(points, K):
    # Project the points to the image plane
    points = (K @ points.T).T
    points = points[:, :3] / points[:, -1:]
    return points[:, :2]
points2d = project_point_cloud(points3d, K)

# Visualize the projected point cloud
plt.scatter(points2d[:, 0], points2d[:, 1])
plt.show()

# Define a 2D depth image
depth = np.zeros((480, 640))
depth_points = []
for i in range(points2d.shape[0]):
    x, y = int(points2d[i, 0]), int(points2d[i, 1])
    if x < 0 or x >= 640 or y < 0 or y >= 480: continue
    if depth[y, x] < points3d[i, 2]:
        depth[y, x] = points3d[i, 2]
        depth_points.append(points3d[i])
depth_points = np.array(depth_points)

# Visualize the depth image
plt.imshow(depth)
plt.show()

print(f"out of {points3d.shape[0]} points, {depth_points.shape[0]} points are in the depth image")


# Center the depth point cloud
t_offset = np.mean(depth_points, axis=0)
depth_points -= t_offset

print(f"The mean is {t_offset} but the original offset is {T[:3, 3]}")
print(f"Now the goal is to sample transforms and try to learn how to recover original offset.")
