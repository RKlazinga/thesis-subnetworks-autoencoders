import os
import torch
from PIL import ImageDraw, Image

SPACING = 200
CHANNEL_SIZE = 40, 20
ON_CHANNEL = (0, 200, 0)
OFF_CHANNEL = (200, 0, 0)


def get_square_coords(bn_idx, single_idx, im_height, bn_mask):
    x = SPACING + bn_idx * (CHANNEL_SIZE[0] + SPACING)
    y = im_height/2 + (SPACING + (single_idx - len(bn_mask) / 2) * (CHANNEL_SIZE[1]+2))
    return x, y


def mask_to_png(mask):
    mask = torch.load(mask)
    bn_count = len(mask)
    most_channels = max(map(len, mask))

    im_width = bn_count * CHANNEL_SIZE[0] + (2 + bn_count) * SPACING
    im_height = most_channels * (CHANNEL_SIZE[1] + 1) + 2 * SPACING
    bg: Image.Image = Image.new("RGB", (im_width, im_height), color=(255, 255, 255))
    draw = ImageDraw.ImageDraw(bg)

    for bn_idx, bn_mask in enumerate(mask):
        for single_idx, single_mask in enumerate(bn_mask):
            x, y = get_square_coords(bn_idx, single_idx, im_height, bn_mask)
            draw.rectangle((x, y, x + CHANNEL_SIZE[0], y + CHANNEL_SIZE[1]), fill=ON_CHANNEL if single_mask else OFF_CHANNEL)

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

    bg.show()


if __name__ == '__main__':
    run_id = "8c6031ee8"

    masks = list(filter(lambda x: x.startswith("keep"), os.listdir(f"runs/{run_id}")))
    masks = [f"runs/{run_id}/{x}" for x in masks]

    m = masks[150]
    mask_to_png(m)
