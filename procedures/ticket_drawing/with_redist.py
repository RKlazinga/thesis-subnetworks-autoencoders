from typing import Dict

import torch
from torch.nn import Linear
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.conv import _ConvNd, _ConvTransposeNd
from torch.nn import functional

from evaluation.pruning_vis import mask_to_png
from models.conv_ae import ConvAE
from procedures.ticket_drawing.without_redist import find_channel_mask_no_redist
from utils.file import change_working_dir
from utils.get_run_id import last_run


def find_channel_mask_redist(network, fraction, redist_function="weightsim"):
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

    # find the 'operator' (conv, transpose conv, linear) that comes after each batchnorm
    modules = [x for x in network.modules() if isinstance(x, (_ConvNd, _ConvTransposeNd, Linear, _BatchNorm))]
    op_after_bn = {bn: modules[modules.index(bn)+1] for bn in bn_masks.keys()}

    total_channels = 0
    bn_of_index = {}
    indices_of_bn = {}
    range_of_index = {}
    idx = 0
    for bn in bn_masks.keys():
        bn_size = bn.weight.data.shape[0]
        total_channels += bn.weight.data.shape[0]
        indices_of_bn[bn] = (idx, idx+bn_size)
        for i in range(idx, idx+bn_size):
            bn_of_index[i] = bn
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
            assert all_remaining_weights < 1e7
            assert 1 < multiplier < 2, f"Multiplier {multiplier} has strange value " \
                                       f"({min_weight}, {all_weights[range_start:range_end]})"
            # scale range up proportionally
            all_weights[range_start:range_end] *= multiplier
        elif redist_function.startswith("weightsim"):
            range_start, range_end = range_of_index[min_idx]
            # scale all_weights[range_start:range_end] in some way

            # get the operator that uses this batch norm's channels
            operator = op_after_bn[bn_of_index[min_idx]]

            assert isinstance(operator, (_ConvNd, _ConvTransposeNd, Linear))

            # get the channel number in local terms (within this layer)
            min_channel = min_idx - range_start

            # get the weight matrix slice corresponding to the channel
            weights_per_channel = {}
            for j in range(range_start, range_end):
                if isinstance(operator, Linear):
                    weights = operator.weight.data[:, j - range_start].clone()
                elif isinstance(operator, _ConvTransposeNd):
                    weights = operator.weight.data[j - range_start, :, :, :].clone()
                else:
                    weights = operator.weight.data[:, j - range_start, :, :].clone()
                # weights *= 1  / torch.sum(weights)

                if "-" in redist_function:
                    eps = float(redist_function.split("-")[-1])
                    mask = torch.lt(torch.abs(weights), eps)
                    weights[mask] = 0
                weights = torch.sign(weights)
                weights_per_channel[j - range_start] = weights
            min_channel_weights = weights_per_channel[min_channel]

            # for all remaining weights, get that slice and compare
            similarities = {}
            for j in range(range_start, range_end):
                if combined_mask[j]:
                    comparison_channel = j - range_start
                    comparison_weights = weights_per_channel[comparison_channel]

                    # compare weights using mean-squared
                    difference = (functional.mse_loss(min_channel_weights, comparison_weights).item() + 1e-5) ** 3
                    similarities[j] = 1 / difference

            # assign the remaining channels a portion of this weight, proportional to similarity
            sim_sum = sum(similarities.values())

            for j in range(range_start, range_end):
                if combined_mask[j]:
                    all_weights[j] += min_weight * similarities[j] / sim_sum

            all_remaining_weights = torch.sum(all_weights[range_start:range_end] *
                                              combined_mask[range_start:range_end])
            # check if we accidentally caught any pruned (1e9) weights
            assert all_remaining_weights < 1e7

        else:
            raise ValueError(f"Unknown redistribution function: {redist_function}")

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
    _network.load_state_dict(torch.load(f"runs/{last_run()}/trained-8.pth"))

    # redist_prop_masks = list(find_channel_mask_redist(_network, 0.5, redist_function="proportional").values())
    redist_sim_masks = list(find_channel_mask_redist(_network, 0.5, redist_function="weightsim").values())
    redist_sim_masks2 = list(find_channel_mask_redist(_network, 0.5, redist_function="weightsim-0.001").values())
    no_redist_masks = list(find_channel_mask_no_redist(_network, 0.5).values())
    # mask_to_png(redist_prop_masks, "Proportionally redistributed weights")
    mask_to_png(redist_sim_masks, "Similarity-based redistributed weights")
    mask_to_png(redist_sim_masks2, "Similarity-based 0.001")
    mask_to_png(no_redist_masks, "No weight redistribution")
