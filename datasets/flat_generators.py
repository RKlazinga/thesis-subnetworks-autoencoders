import math

import torch

from datasets.im_generators import r_float


def random_sine_gaussian(size=16):
    def sine(a, b, x):
        return a * math.sin(math.radians((x - b) * 22)) + 5
    # gaussian noise per pixel, centered on a sign function
    a = r_float(1, 2)
    b = r_float(16)

    mean = torch.tensor([sine(a, b, x) for x in range(size)])
    std = torch.full((size, ), 0.5)
    return torch.normal(mean, std).float()


def flat_gaussian(size=16):
    # gaussian noise per pixel, centered on a function
    mean = torch.full((size, ), 5, dtype=torch.float)
    std = torch.full((size, ), 2)
    return torch.normal(mean, std).float()
