import random

import torch


def generate_random_str():
    return hex(random.randint(16**8, 16**9))[2:]


__DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def dev():
    return __DEVICE
