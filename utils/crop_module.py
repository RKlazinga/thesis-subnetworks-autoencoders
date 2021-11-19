from torch import nn
from torchvision.transforms.functional import center_crop


class Crop(nn.Module):
    def __init__(self, desired_size):
        super().__init__()
        self.size = desired_size

    def forward(self, x):
        return center_crop(x, self.size)