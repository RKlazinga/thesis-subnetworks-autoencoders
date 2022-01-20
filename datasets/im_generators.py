import math
import random

from PIL import Image, ImageDraw


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


def dummy_gen():
    bg = Image.new("L", (30, 30))
    draw = ImageDraw.Draw(bg)

    is_filled = random.random() > 0.5

    a = random.randint(1, 50)
    b = random.randint(0, 30)
    c = random.randint(0, 30)
    d = -1 if random.random() > 0.5 else 1
    e = random.randint(1, 5)
    f = random.randint(-5, 20)

    points = []
    for i in range(a):
        x = b + (f - i/3) * math.sin(i / e * d)
        y = c + (f - i/3) * math.cos(i / e * d)
        points.append((x, y))

    draw.line(points, fill=255, width=2)#, joint="curve")

    return bg


if __name__ == '__main__':
    dummy_gen()
