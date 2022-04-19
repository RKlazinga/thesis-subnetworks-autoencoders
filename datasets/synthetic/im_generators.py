import math

import torch
from PIL import Image, ImageFont, ImageDraw
from torchvision.transforms import ToPILImage

from datasets.synthetic.common import r_float
from models.conv_ae import ConvAE
from settings.data_settings import NORMAL_STD_DEV, NUM_VARIABLES, CHANNEL_STACKING
from utils.file import change_working_dir


def stacked_sine2d():
    return torch.concat([sine_2d() for _ in range(CHANNEL_STACKING)], dim=0)


def sine_2d(std=NORMAL_STD_DEV, num_variables=NUM_VARIABLES):
    im_size = 28

    def sine(y, x, fx=40, fy=40, a=0.5, p=0, sx=0, sy=0, bx=im_size//2, by=im_size//2):
        # circle
        # r = (((x-bx) * fx) ** 2 + ((y-by) * fy) ** 2) ** 0.5 + p
        # horizontal
        # r = ((x-bx) * fx) + p
        # diagonal
        # r = ((x-bx) * fx + (y-by) * fy) + p
        # square
        r = max((abs(x-bx + sx*(y-by))) * fx, (abs(y-by + sy*(x-bx))) * fy) + p
        return a * math.sin(math.radians(r)) + 0.5
    variables = [
        r_float(30, 100),  # x frequency
        r_float(30, 100),  # y frequency
        r_float(0.3, 0.8),  # amplitude
        r_float(360),  # phase
        r_float(-0.5, 0.5),  # skew x
        r_float(-0.5, 0.5),  # skew y
        r_float(4, im_size-4),  # x offset
        r_float(4, im_size-4),  # y offset
    ][:num_variables]
    # gaussian noise per pixel, centered on a sine function

    mean = torch.tensor([sine(x, y, *variables) for x in range(im_size) for y in range(im_size)]).view((im_size, im_size))
    std = torch.full((im_size, im_size), std)
    return torch.clamp(torch.normal(mean, std).float(), 0, 1).view((1, im_size, im_size))


if __name__ == '__main__':
    height = 8
    dims = 5
    rim = 8
    spacing = (2, 12)
    size = 28
    font = ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", size=14)
    textsize = font.getsize("D=1xx")
    bg = Image.new("L", (height*28 + (height-1) * spacing[0] + rim + textsize[0], dims*size + 2 * rim + (dims-1) * spacing[1]), color=0)
    draw = ImageDraw.Draw(bg)
    for d in range(dims):
        for i in range(height):
            im = ToPILImage()(sine_2d(0.01, d))
            bg.paste(im, (i*(size + spacing[0]) + textsize[0], rim + d*(size + spacing[1])))
        text = f"D={d}"
        w = font.getsize(text)[0]
        draw.text(((textsize[0] - w) // 2, rim + d*(size + spacing[1]) + textsize[1] // 2), text=text, font=font, fill=255)
    change_working_dir()
    bg.save("figures/synth_2d_ex.png")
    # bg.show()
    # ToPILImage()(sine_2d(0.01, 2)).show()
    # ToPILImage()(sine_2d(0.01, 3)).show()
    # ToPILImage()(sine_2d(0.01, 3)).show()
    # ToPILImage()(sine_2d(0.01, 4)).show()
    # ToPILImage()(sine_2d(0.01, 4)).show()