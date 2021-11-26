import torch
import torchsummary
from torch.nn import Linear, Unflatten
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.conv import _ConvTransposeNd, _ConvNd


def prune_model(model, masks):
    next_iter = model.modules()
    next(next_iter)

    mask_idx = -1
    out_mask = None

    for m, m_next in zip(model.modules(), next_iter):
        if any([isinstance(m, x) for x in [_ConvNd, _ConvTransposeNd, Linear]]) and isinstance(m_next, _BatchNorm):
            mask_idx += 1
            in_mask = out_mask
            out_mask = masks[mask_idx].bool()
            is_transpose = int(isinstance(m, _ConvTransposeNd))

            if in_mask is not None:
                # prune expected input
                prune_parameter(m, "weight", in_mask, axis=1-is_transpose)

            # prune output
            prune_parameter(m, "weight", out_mask, axis=is_transpose)
            prune_parameter(m, "bias", out_mask, axis=0)

            # prune expected input of batchnorm (using out_mask, since batchnorm comes after)
            prune_parameter(m_next, "weight", out_mask, axis=0)
            prune_parameter(m_next, "bias", out_mask, axis=0)
            prune_parameter(m_next, "running_mean", out_mask, axis=0)
            prune_parameter(m_next, "running_var", out_mask, axis=0)

        elif isinstance(m, Unflatten):
            # if we encounter an unflatten layer, reset the expected input mask to all 1's of the appropriate size
            out_mask = torch.ones(m.unflattened_size[0]).bool()

        elif isinstance(m, Linear):
            # unmasked linear layer still needs to account for pruning in previous layers
            prune_parameter(m, "weight", out_mask, axis=1)


def prune_parameter(module, parameter_name, mask, axis=0):
    """
    Prune a single parameter Tensor within a module
    :param module:
    :param parameter_name:
    :param mask:
    :param axis:
    :return:
    """
    param = getattr(module, parameter_name)
    if param is not None:
        n = param.data.shape[axis]
        if isinstance(module, Linear):
            # repeat the mask accordingly (because the channels were flattened)
            repeat_ratio = int(n / torch.numel(mask))
            mask = torch.repeat_interleave(mask.clone(), repeat_ratio)

        keep_indices = torch.arange(n)[mask]
        param.data = param.data.index_select(axis, keep_indices)
        if param.grad is not None:
            param.grad.data = param.grad.data.index_select(axis, keep_indices)
