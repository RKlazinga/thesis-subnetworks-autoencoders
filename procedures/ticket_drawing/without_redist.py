from typing import Dict

import torch
from torch.nn.modules.batchnorm import _BatchNorm


def mask_dist(masks_a, masks_b):
    return [torch.sum(torch.abs(a-b)).item() for a, b in zip(masks_a, masks_b)]


def find_channel_mask_no_redist(network, fraction):
    """
    Draw a critical subnetwork from a given trained network using channel pruning.
    Pruning is applied layer-by-layer.

    :param network: The trained network to draw from
    :param fraction: The fraction of channels to KEEP after pruning
    :return: A mask for each BatchNorm layer
    """
    bn_masks: Dict[_BatchNorm, torch.Tensor] = {
        m: None for m in network.modules() if isinstance(m, _BatchNorm)
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
