import math

import torch
from torchvision.transforms import ToPILImage

from datasets.synthetic.common import r_float
from models.conv_ae import ConvAE
from settings.data_settings import NORMAL_STD_DEV, NUM_VARIABLES


def sine_2d(std=NORMAL_STD_DEV, num_variables=NUM_VARIABLES):
    im_size = ConvAE.IMAGE_SIZE

    def sine(x, y, fx=40, fy=40, a=0.5, p=0, bx=im_size//2, by=im_size//2):
        # circle
        # r = (((x-bx) * fx) ** 2 + ((y-by) * fy) ** 2) ** 0.5 + p
        # horizontal
        # r = ((x-bx) * fx) + p
        # diagonal
        # r = ((x-bx) * fx + (y-by) * fy) + p
        # square
        r = max((abs(x-bx)) * fx, (abs(y-by)) * fy) + p
        return a * math.sin(math.radians(r)) + 0.5
    variables = [
        r_float(30, 100),  # x frequency
        r_float(30, 100),  # y frequency
        r_float(0.4, 0.7),  # amplitude
        r_float(360),  # phase
        r_float(4, im_size-4),  # x offset
        r_float(4, im_size-4),  # y offset
    ][:num_variables]
    # gaussian noise per pixel, centered on a sine function

    mean = torch.tensor([sine(x, y, *variables) for x in range(im_size) for y in range(im_size)]).view((im_size, im_size))
    std = torch.full((im_size, im_size), std)
    return torch.clamp(torch.normal(mean, std).float(), 0, 1).view((1, im_size, im_size))


if __name__ == '__main__':
    ToPILImage()(sine_2d(0.01, 4)).show()
    ToPILImage()(sine_2d(0.01, 4)).show()
    ToPILImage()(sine_2d(0.01, 4)).show()
