import matplotlib.pyplot as plt
import numpy as np

def visualize_point_cloud(points, colors=None, show=True):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors)
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    if show: plt.show()


def viz_image_batch(images, block=True, title=None):
    B = images.shape[0]
    rows = int(np.floor(np.sqrt(B)))
    cols  = int(np.ceil(B / rows))
    fig, axs = plt.subplots(rows, cols)
    for i in range(B):
        r, c = i // cols, i % cols
        axs[r][c].imshow(images[i])
    plt.show(block=block)
    if title is not None: fig.suptitle(title)
    return fig