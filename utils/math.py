import math


def calculate_im_size(im_size, pooling):
    """
    Determine the channel count of a given NxN image after X max-pool operators with stride=2 and ceil_mode=True
    """
    for i in range(pooling):
        im_size = math.ceil(im_size / 2)
    return int(im_size)
