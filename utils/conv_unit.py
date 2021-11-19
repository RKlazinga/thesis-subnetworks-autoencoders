from torch import nn

from utils.batchnorm_filter import FilteredBatchNorm


class ConvUnit(nn.Module):
    """
    Basic combination of convolution, normalisation and activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, *args,
                 activation=nn.ReLU, max_pool=False, bn_mask=None, **kwargs):
        super().__init__()

        self.steps = [
            nn.Conv2d(in_channels, out_channels, kernel_size, *args, **kwargs),
        ]

        if bn_mask is not False:
            self.steps.append(FilteredBatchNorm(out_channels, bn_mask))

        self.steps.append(activation())

        if max_pool:
            self.steps.append(nn.MaxPool2d(2, ceil_mode=True))

        self.steps = nn.Sequential(*self.steps)

    def forward(self, x):
        return self.steps(x)


class ConvTransposeUnit(nn.Module):
    """
    Basic combination of convolution, normalisation and activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, *args,
                 activation=nn.ReLU, max_pool=False, bn_mask=None, **kwargs):
        super().__init__()

        self.steps = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, *args, **kwargs),
        ]

        if bn_mask is not False:
            self.steps.append(FilteredBatchNorm(out_channels, bn_mask))

        self.steps.append(activation())

        if max_pool:
            self.steps.append(nn.MaxPool2d(2, ceil_mode=True))

        self.steps = nn.Sequential(*self.steps)

    def forward(self, x):
        return self.steps(x)
