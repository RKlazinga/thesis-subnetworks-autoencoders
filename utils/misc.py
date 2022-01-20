import random


def generate_random_str():
    return hex(random.randint(16**8, 16**9))[2:]
