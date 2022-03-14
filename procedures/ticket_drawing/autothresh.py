from typing import Dict

import torch
from torch.nn.modules.batchnorm import _BatchNorm

from models.conv_ae import ConvAE
from settings.global_settings import RUN_FOLDER
from utils.file import change_working_dir
from utils.get_run_id import last_run


def find_channel_mask_autothresh(network, thresh=0.1):
    """
    Draw a critical subnetwork from a given trained network using channel pruning.
    Pruning is applied layer-by-layer.

    :param network: The trained network to draw from
    :param thresh: The treshold below which to prune
    :return: A mask for each BatchNorm layer
    """
    bn_masks: Dict[_BatchNorm, torch.Tensor] = {
        m: None for m in network.modules() if isinstance(m, _BatchNorm)
    }

    if len(bn_masks) > 0:
        for idx, bn in enumerate(bn_masks.keys()):
            weight = bn.weight.data.clone().abs()
            mask = weight.gt(thresh).float()

            # if the mask is all empty, forcefully keep the most important channel to ensure some data can flow through
            if torch.count_nonzero(mask) == 0:
                mask[torch.argmax(weight)] = 1

            bn_masks[bn] = mask

    return bn_masks


if __name__ == '__main__':
    change_working_dir()
    net = ConvAE(6, 4, 6)
    net.load_state_dict(torch.load(f"{RUN_FOLDER}/{last_run()}/trained-3.pth"))

    masks = find_channel_mask_autothresh(net)
    remaining_channels = 0
    total_channels = 0
    for mask in masks.values():
        remaining_channels += torch.count_nonzero(mask).item()
        total_channels += torch.numel(mask)
        print(f"{torch.count_nonzero(mask)}/{torch.numel(mask)} channels retained")

    print(f"Effective pruning ratio {round(remaining_channels/total_channels, 2)}")
