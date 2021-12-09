import torch
from torch.nn import Linear, Unflatten
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.conv import _ConvTransposeNd, _ConvNd


def prune_model(model, masks):
    mask_idx = -1
    out_mask = None

    modules = list(model.modules())
    modules = [x for x in modules if isinstance(x, (_ConvNd, _ConvTransposeNd, Linear, _BatchNorm, Unflatten))]

    for i in range(len(modules)-2):
        operator, b, c = modules[i:i+3]
        if isinstance(operator, (_ConvNd, _ConvTransposeNd, Linear)):
            if isinstance(b, _BatchNorm):
                unflatten = None
                bn = b
            elif isinstance(b, Unflatten) and isinstance(c, _BatchNorm):
                unflatten = b
                bn = c
            else:
                continue

            mask_idx += 1
            in_mask = out_mask
            out_mask = masks[mask_idx].bool()
            is_transpose = int(isinstance(operator, _ConvTransposeNd))

            if in_mask is not None:
                # prune expected input
                prune_parameter(operator, "weight", in_mask, axis=1-is_transpose)

            # adjust unflatten layer if present
            if unflatten:
                current_shape = unflatten.unflattened_size
                unflatten.unflattened_size = (
                    torch.count_nonzero(out_mask).item(),
                    *current_shape[1:]
                )

            # prune output
            prune_parameter(operator, "weight", out_mask, axis=is_transpose)
            prune_parameter(operator, "bias", out_mask, axis=0)

            # prune expected input of batchnorm (using out_mask, since batchnorm comes after)
            prune_parameter(bn, "weight", out_mask, axis=0)
            prune_parameter(bn, "bias", out_mask, axis=0)
            prune_parameter(bn, "running_mean", out_mask, axis=0)
            prune_parameter(bn, "running_var", out_mask, axis=0)


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
