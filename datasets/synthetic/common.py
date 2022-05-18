import random


def r_int(a, b):
    return random.randint(a, b)


def r_bool():
    return random.random() > 0.5


def r_sign():
    return -1 if random.random() > 0.5 else 1


def r_float(a, b=None):
    if b is None:
        b = a
        a = 0

    return random.random() * (b - a) + a
