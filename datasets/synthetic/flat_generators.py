import math

import torch

from datasets.synthetic.common import r_float
from settings import Settings


def random_sine_gaussian(size=None, std=None, num_variables=None):
    # avoid binding default values during function definition in case Settings are changed on the fly
    if size is None:
        size = Settings.FLAT_DATAPOINTS
    if std is None:
        std = Settings.NORMAL_STD_DEV
    if num_variables is None:
        num_variables = Settings.NUM_VARIABLES

    def sine(x, a, b=0, c=0, d=22):
        return a * math.sin(math.radians((x - b) * d)) + c
    variables = [
        r_float(0.1, 1),  # amplitude
        r_float(16),  # phase
        r_float(0, 1),  # bias
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

    for dim, var in zip(range(1, 2), ["amplitude", "phase", "bias", "frequency"]):
        plots = [random_sine_gaussian(16, 0.05, dim).tolist() for _ in range(1)]
        [plt.plot(p) for p in plots]
        plt.title(f"D={dim} (+{var})")
        plt.tight_layout()
        plt.show()