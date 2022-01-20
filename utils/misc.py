import random

import torch


def generate_random_str():
    return hex(random.randint(16**8, 16**9))[2:]


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
