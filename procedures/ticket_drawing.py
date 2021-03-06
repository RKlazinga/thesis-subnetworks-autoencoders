from typing import Dict

import torch
from torch.nn.modules.batchnorm import _BatchNorm

from settings import Settings


def mask_dist(masks_a, masks_b):
    return [torch.sum(torch.abs(a-b)).item() for a, b in zip(masks_a, masks_b)]


def find_channel_mask(network, fraction, per_layer_limit=Settings.PRUNE_LIMIT):
    """
    Draw a critical subnetwork from a given trained network using (global) channel pruning.

    :param network: The trained network to draw from
    :param fraction: The fraction of channels to remove during pruning
    :param per_layer_limit: Do not prune a single layer beyond this limit.
                            Set equal to fraction to apply the pruning per-layer
    :return: A mask for each BatchNorm layer
    """
    bn_masks: Dict[_BatchNorm, torch.Tensor] = {
        m: None for m in network.modules() if isinstance(m, _BatchNorm)
    }

    assert fraction <= per_layer_limit

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

            if per_layer_limit > 0:
                # find the per_layer_limit most important weights in the layer, and set them to ~inf
                layer_weights_sorted = torch.sort(abs_weight).values
                cutoff_idx = min(bn_size-1, round(bn_size * per_layer_limit))
                layer_threshold = layer_weights_sorted[cutoff_idx]

                layer_mask = abs_weight.ge(layer_threshold)
                abs_weight[layer_mask] = 1e9

            all_weights[idx:idx+bn_size] = abs_weight
            idx += bn_size

        # based on the fraction, find the global weight threshold below which we will prune
        # since the sort is ascending, we find the threshold at [the fraction to remove]
        weights_sorted = torch.sort(all_weights).values

        threshold = weights_sorted[int(total_channels * fraction)]

        for idx, bn in enumerate(bn_masks.keys()):
            bn_size = bn.weight.data.shape[0]
            abs_weight = bn.weight.data.clone().abs()

            if per_layer_limit > 0:
                # find the per_layer_limit most important weights in the layer, and set them to ~inf
                layer_weights_sorted = torch.sort(abs_weight).values
                cutoff_idx = min(bn_size-1, round(bn_size * per_layer_limit))
                layer_threshold = layer_weights_sorted[cutoff_idx]

                layer_mask = abs_weight.ge(layer_threshold)
                abs_weight[layer_mask] = 1e9

            mask = abs_weight.ge(threshold).float()
            bn_masks[bn] = mask

    return bn_masks
