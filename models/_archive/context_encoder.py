import torch
from torch import nn
from torch import Tensor
from torch.nn import Module, Conv2d
from torchvision.models import resnet18

from torchvision.transforms import Resize
from torchvision.models.segmentation import deeplabv3_resnet50

class CtxtEncoder(Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.in_dim, self.out_dim = in_dim, out_dim
        self._net = resnet18()
        self._net.conv1 = Conv2d(in_dim, 64, (7, 7), (2, 2), (3, 3), bias=False)
        self._net.fc = nn.Linear(512, out_dim)

    def forward(self, x):
        return self._net(x)
