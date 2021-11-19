import torch
from torch import nn


class FilteredBatchNorm(nn.Module):
    def __init__(self, out_channels, mask=None):
        super().__init__()
        self.bn = nn.BatchNorm2d(out_channels)

        if mask is None:
            mask = torch.ones(out_channels)
        self.mask = mask.bool()

    def forward(self, x):
        return self.bn(x)[:, self.mask, :, :]
