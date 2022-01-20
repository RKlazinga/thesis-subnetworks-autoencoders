from typing import Union

import torch
from PIL import ImageDraw, ImageFont, Image

from utils.file import change_working_dir
from utils.get_run_id import last_run

HEADER = 50
SPACING = 150
CHANNEL_SIZE = 40, 20
ON_CHANNEL = (0, 200, 0)
OFF_CHANNEL = (200, 0, 0)


def get_square_coords(bn_idx, single_idx, im_height, bn_mask):
    x = SPACING + bn_idx * (CHANNEL_SIZE[0] + SPACING)
    y = (im_height - HEADER)/2 + (single_idx - len(bn_mask) / 2) * (CHANNEL_SIZE[1]+2) + 1 + HEADER
    return x, y


def mask_to_png(mask: Union[str, list[torch.Tensor]], caption=None, draw_conn=False):
    print(mask)
    if type(mask) == str:
        mask = torch.load(mask, map_location=torch.device('cpu'))
    elif type(mask) != list:
        raise TypeError(f"Input mask has unexpected type {type(mask)}")

    bn_count = len(mask)
    most_channels = max(map(len, mask))

    im_width = bn_count * CHANNEL_SIZE[0] + (1 + bn_count) * SPACING
    im_height = most_channels * (CHANNEL_SIZE[1] + 2) + SPACING - 1 + HEADER
    bg: Image.Image = Image.new("RGB", (im_width, im_height), color=(255, 255, 255))
    draw = ImageDraw.ImageDraw(bg)

    for bn_idx, bn_mask in enumerate(mask):
        for single_idx, single_mask in enumerate(bn_mask):
            x, y = get_square_coords(bn_idx, single_idx, im_height, bn_mask)
            draw.rectangle((x, y, x + CHANNEL_SIZE[0], y + CHANNEL_SIZE[1]), fill=ON_CHANNEL if single_mask else OFF_CHANNEL)

    if draw_conn:
        for bn_idx, bn_mask_left in enumerate(mask):
            if bn_idx < bn_count - 1:
                bn_mask_right = mask[bn_idx + 1]

                for left_idx, left in enumerate(bn_mask_left):
                    if left:
                        left_x, left_y = get_square_coords(bn_idx, left_idx, im_height, bn_mask_left)
                        left_x += CHANNEL_SIZE[0]
                        left_y += CHANNEL_SIZE[1] / 2
                        for right_idx, right in enumerate(bn_mask_right):
                            if right:
                                right_x, right_y = get_square_coords(bn_idx + 1, right_idx, im_height, bn_mask_right)
                                right_y += CHANNEL_SIZE[1] / 2
                                draw.line(((left_x, left_y), (right_x, right_y)), fill=ON_CHANNEL, width=2)

    if caption:
        font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", size=60)
        fwidth = font.getsize(caption)[0]
        draw.text(((im_width - fwidth) // 2, 30), caption, (0, 0, 0), font=font)

    # bg.show()
    bg.save(f"graphs/pruning-vis-{caption}.png")


if __name__ == '__main__':
    change_working_dir()
    mask_to_png(torch.load(f"runs/bnreg0-[6, 4, 6]-b3d0a0d4c/keep-0.7-epoch-4-4.pth"), "No BN regularisation")
    mask_to_png(torch.load(f"runs/bnreg0.0001-[6, 4, 6]-9e2153967/keep-0.7-epoch-4-4.pth"), "Reg=1e-4 (EB default)")
    mask_to_png(torch.load(f"runs/bnreg0.001-[6, 4, 6]-eed07da1d/keep-0.7-epoch-4-4.pth"), "Reg=1e-3")
    # mask_to_png(torch.load(f"runs/{last_run()}/keep-0.5-epoch-15-4.pth"))

