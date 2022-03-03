import math

import torch
from torchvision.transforms import ToPILImage

from datasets.synthetic.common import r_float
from models.conv_ae import ConvAE
from settings.data_settings import NORMAL_STD_DEV, NUM_VARIABLES


def sine_2d(std=NORMAL_STD_DEV, num_variables=NUM_VARIABLES):
    im_size = ConvAE.IMAGE_SIZE

    def sine(x, y, a, p=0, bx=im_size//2, fx=40, by=im_size//2, fy=40):
        r = (((x-bx) * fx) ** 2 + ((y-by) * fy) ** 2) ** 0.5 + p
        return a * math.sin(math.radians(r)) + 0.5
    # gaussian noise per pixel, centered on a sign function
    variables = [
        r_float(0.4, 0.6),  # amplitude
        r_float(360),  # phase
        r_float(4, im_size-4),  # x offset
        r_float(30, 100),  # x frequency
        r_float(4, im_size-4),  # y offset
        r_float(30, 100),  # y frequency
    ][:num_variables]

    mean = torch.tensor([sine(x, y, *variables) for x in range(im_size) for y in range(im_size)]).view((im_size, im_size))
    std = torch.full((im_size, im_size), std)
    return torch.clamp(torch.normal(mean, std).float(), 0, 1)


if __name__ == '__main__':
    ToPILImage()(sine_2d(0.01, 1)).show()
    ToPILImage()(sine_2d(0.01, 2)).show()
    ToPILImage()(sine_2d(0.01, 3)).show()
    ToPILImage()(sine_2d(0.01, 4)).show()
    ToPILImage()(sine_2d(0.01, 5)).show()
    ToPILImage()(sine_2d(0.01, 6)).show()
