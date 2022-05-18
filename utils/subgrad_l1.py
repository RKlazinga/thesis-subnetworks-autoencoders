# channel sparsity regularisation as introduced by Liu et al. 2017
# taken from https://github.com/RICE-EIC/Early-Bird-Tickets/blob/4a16ae0731edf288c48e000c1c2a51dc0433f4ef/main.py#L234
# "additional subgradient descent on the sparsity-induced penalty term"
import torch
from torch.nn.modules.batchnorm import _BatchNorm, BatchNorm1d

from settings import Settings


def update_bn(model):
    for m in model.modules():
        if isinstance(m, _BatchNorm):
            if isinstance(m, BatchNorm1d):
                if m.weight.data.shape[0] == Settings.TOPOLOGY[0]:
                    m.weight.grad.data.add_(
                        Settings.REG_MULTIPLIER * Settings.LATENT_SPARSITY_PENALTY * torch.sign(m.weight.data)
                    )
                    # set weights below a threshold to 0
                    mask = torch.abs(m.weight.data) < 1e-4
                    m.weight.data[mask] = 0
                    m.weight.grad.data[mask] = 0
                else:
                    m.weight.grad.data.add_(
                        Settings.REG_MULTIPLIER * Settings.LINEAR_SPARSITY_PENALTY * torch.sign(m.weight.data)
                    )
            else:
                m.weight.grad.data.add_(Settings.REG_MULTIPLIER * Settings.CONV_SPARSITY_PENALTY * torch.sign(m.weight.data))
