import math

import torch

from datasets.synthetic.common import r_float
from settings import Settings

im_size = 28


def stacked_sine2d():
    return torch.concat([sine_2d() for _ in range(Settings.CHANNEL_STACKING)], dim=0)


def sine(y, x, fx=40, fy=40, a=0.5, p=0, sx=0, sy=0, bx=im_size//2, by=im_size//2):
    # square
    r = max((abs(x-bx + sx*(y-by))) * fx, (abs(y-by + sy*(x-bx))) * fy) + p
    return a * math.sin(math.radians(r)) + 0.5


def sine_2d():
    variables = [
        r_float(30, 100),  # x frequency
        r_float(30, 100),  # y frequency
        r_float(0.25, 1),  # amplitude
        r_float(360),  # phase
        r_float(-0.5, 0.5),  # skew x
        r_float(-0.5, 0.5),  # skew y
        r_float(4, im_size-4),  # x offset
        r_float(4, im_size-4),  # y offset
    ][:Settings.NUM_VARIABLES]

    # gaussian noise per pixel, centered on a sine function
    mean = torch.tensor([sine(x, y, *variables) for x in range(im_size) for y in range(im_size)]).view((im_size, im_size))
    std = torch.full((im_size, im_size), Settings.NORMAL_STD_DEV)
    return torch.clamp(torch.normal(mean, std).float(), 0, 1).view((1, im_size, im_size))


