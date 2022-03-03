import math

import torch

from datasets.synthetic.common import r_float
from settings.data_settings import NORMAL_STD_DEV, FLAT_DATAPOINTS, NUM_VARIABLES


def random_sine_gaussian(size=FLAT_DATAPOINTS, std=NORMAL_STD_DEV, num_variables=NUM_VARIABLES):
    def sine(x, a, b=0, c=5, d=22):
        return a * math.sin(math.radians((x - b) * d)) + c
    # gaussian noise per pixel, centered on a sign function
    variables = [
        r_float(1, 5),  # amplitude
        r_float(16),  # phase
        r_float(5, 10),  # bias
        r_float(11, 33)  # frequency
    ][:num_variables]

    mean = torch.tensor([sine(x, *variables) for x in range(size)])
    std = torch.full((size, ), std)
    return torch.normal(mean, std).float()


def flat_gaussian(size=16):
    # gaussian noise per pixel, centered on a function
    mean = torch.full((size, ), 5, dtype=torch.float)
    std = torch.full((size, ), 2)
    return torch.normal(mean, std).float()


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    plt.plot(random_sine_gaussian().tolist())
    plt.tight_layout()
    plt.show()