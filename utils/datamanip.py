from scipy.spatial.transform import Rotation
import numpy as np


def project_point_cloud(points, K):
    # Project the points to the image plane
    points = (K @ points.T).T
    points = points[:, :3] / points[:, -1:]
    return points[:, :2]



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





#### TODO: REMOVE 

import matplotlib.pyplot as plt

def visualize_point_cloud(points, colors=None, show=True):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors)
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    if show: plt.show()

