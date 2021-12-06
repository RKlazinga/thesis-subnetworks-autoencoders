from typing import Dict

import torch
from torch.nn.modules.batchnorm import _BatchNorm

from evaluation.pruning_vis import mask_to_png
from models.conv_ae import ConvAE
from procedures.ticket_drawing.without_redist import find_channel_mask_no_redist
from utils.ensure_correct_folder import change_working_dir


def find_channel_mask_redist(network, fraction, redist_function="proportional"):
    """
    Draw a critical subnetwork from a given trained network using channel pruning.
    Pruning is applied layer-by-layer.

    :param network: The trained network to draw from
    :param fraction: The fraction of channels to KEEP after pruning
    :param redist_function: How to redistribute the weight of a pruned channel
    :return: A mask for each BatchNorm layer
    """
    bn_masks: Dict[_BatchNorm, torch.Tensor] = {
        m: None for m in network.modules() if isinstance(m, _BatchNorm)
    }

    if len(bn_masks) == 0:
        return bn_masks

    total_channels = 0
    indices_of_bn = {}
    range_of_index = {}
    idx = 0
    for bn in bn_masks.keys():
        bn_size = bn.weight.data.shape[0]
        total_channels += bn.weight.data.shape[0]
        indices_of_bn[bn] = (idx, idx+bn_size)
        for i in range(idx, idx+bn_size):
            range_of_index[i] = (idx, idx+bn_size)
        idx += bn_size

    # load all batch_norm weights into a single tensor
    combined_mask = torch.ones(total_channels)
    all_weights = torch.zeros(total_channels)
    idx = 0
    for bn in bn_masks.keys():
        bn_size = bn.weight.data.shape[0]

        # set the corresponding section of the all_weights tensor
        abs_weight = bn.weight.data.abs().clone()
        if bn_size == 1:
            abs_weight += 1e9
        all_weights[idx:idx+bn_size] = abs_weight
        idx += bn_size

    # until we reach the desired pruning ratio, remove one channel at a time and redistribute its weight proportionally
    desired_prune_count = int(total_channels * (1 - fraction))

    for i in range(desired_prune_count):
        # find the smallest weight
        min_idx = torch.argmin(all_weights).item()
        min_weight = all_weights[min_idx]

        # add the channel to the mask
        combined_mask[min_idx] = 0

        # redistribute the weight over the relevant range
        if redist_function == "proportional":
            range_start, range_end = range_of_index[min_idx]

            # multiplier = 1 + (weight to remove relative to all other remaining weights)
            all_remaining_weights = torch.sum(all_weights[range_start:range_end] *
                                              combined_mask[range_start:range_end])
            # check if we accidentally caught any pruned (1e9) weights
            multiplier = 1 + min_weight / all_remaining_weights
            assert all_remaining_weights < 1e9
            assert 1 < multiplier < 2, f"Multiplier {multiplier} has strange value ({min_weight}, {all_weights[range_start:range_end]})"
            # scale range up proportionally
            all_weights[range_start:range_end] *= multiplier

        # make sure this weight doesn't get caught by argmin again
        all_weights[min_idx] = 1e9

    # fill bn_masks
    for bn in bn_masks.keys():
        range_start, range_end = indices_of_bn[bn]
        bn_masks[bn] = combined_mask[range_start:range_end].clone()

    return bn_masks


if __name__ == '__main__':
    change_working_dir()
    _network = ConvAE(6, 4, 6)
    _network.load_state_dict(torch.load("runs/1ed4ec56d/trained-1.pth"))

    redist_masks = list(find_channel_mask_redist(_network, 0.5).values())
    no_redist_masks = list(find_channel_mask_no_redist(_network, 0.5).values())
    mask_to_png(redist_masks, "Proportionally redistributed weights")
    mask_to_png(no_redist_masks, "No weight redistribution")

