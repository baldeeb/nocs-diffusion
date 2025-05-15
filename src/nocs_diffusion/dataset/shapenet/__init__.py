
DEFAULT_SPLIT_PERCENTAGES = {
    'train': 0.7,
    'val': 0.15,
    'test': 0.15
}

from .dataloader import ShapeNetDataloader
from .dataset import ShapeNetDataset
from ..utils.mesh_renderer.renderer import MeshRenderer
