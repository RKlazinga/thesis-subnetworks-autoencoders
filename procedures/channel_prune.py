from typing import Dict

import torch
from torch import nn

from models.conv_ae import ConvAE


def mask_dist(masks_a, masks_b):
    return [torch.sum(torch.abs(a-b)).item() for a, b in zip(masks_a, masks_b)]


def find_channel_mask(network, fraction):
    """
    Draw a critical subnetwork from a given trained network using channel pruning.
    Pruning is applied layer-by-layer.

    :param network: The trained network to draw from
    :param fraction: The fraction of channels to KEEP after pruning
    :return: A mask for each BatchNorm layer
    """
    bn_masks: Dict[nn.BatchNorm2d, torch.Tensor] = {
        m: None for m in network.modules() if isinstance(m, nn.BatchNorm2d)
    }

    if len(bn_masks) > 0:
        total_channels = 0
        for bn in bn_masks.keys():
            total_channels += bn.weight.data.shape[0]

        # load all batch_norm weights into a single tensor
        all_weights = torch.zeros(total_channels)
        idx = 0
        for bn in bn_masks.keys():
            bn_size = bn.weight.data.shape[0]

            # set the corresponding section of the all_weights tensor
            abs_weight = bn.weight.data.abs().clone()
            all_weights[idx:idx+bn_size] = abs_weight
            idx += bn_size

        # based on the fraction, find the global weight threshold below which we will prune
        # since the sort is ascending, we find the threshold at (1-[the fraction to keep])
        weights_sorted = torch.sort(all_weights).values

        threshold = weights_sorted[int(total_channels * (1 - fraction))]

        for idx, bn in enumerate(bn_masks.keys()):
            weight = bn.weight.data.clone().abs()
            mask = weight.gt(threshold).float()

            # if the mask is all empty, forcefully keep the most important channel to ensure some data can flow through
            if torch.count_nonzero(mask) == 0:
                mask[torch.argmax(weight)] = 1

            bn_masks[bn] = mask

    return bn_masks

    # per batch layer:
    #
    # m.weight.data.mul_(mask)
    # m.bias.data.mul_(mask)
    #
    # definitively remove channels with "channel_selection" module:
    # https://github.com/RICE-EIC/Early-Bird-Tickets/blob/4a16ae0731edf288c48e000c1c2a51dc0433f4ef/models/channel_selection.py#L7


if __name__ == '__main__':
    network = ConvAE(6, 4, 6)
    network.load_state_dict(torch.load("network.pth"))

    masks = find_channel_mask(network, 0.5)
    for m in masks.values():
        print(f"{torch.count_nonzero(m)} of {m.numel()} layers kept ({round(float(torch.count_nonzero(m)/m.numel())*100, 1)}%))")
