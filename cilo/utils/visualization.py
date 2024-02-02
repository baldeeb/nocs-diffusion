import matplotlib.pyplot as plt
import numpy as np

def visualize_point_cloud(points, colors=None, show=True):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors)
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    if show: plt.show()


def viz_image_batch(images):
    B = images.shape[0]
    cols = int(np.floor(np.sqrt(B)))
    rows  = int(np.ceil(B / cols))
    fig, axs = plt.subplots(rows, cols)
    for i in range(B):
        r, c = i // cols, i % cols
        axs[r][c].imshow(images[i])
    plt.show()
    return fig