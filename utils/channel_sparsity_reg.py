# channel sparsity regularisation as introduced by Liu et al. 2017
# taken from https://github.com/RICE-EIC/Early-Bird-Tickets/blob/4a16ae0731edf288c48e000c1c2a51dc0433f4ef/main.py#L277
# "additional subgradient descent on the sparsity-induced penalty term"
import torch
from torch.nn.modules.batchnorm import _BatchNorm, BatchNorm1d

from settings.s import Settings


def update_bn(model, conv_penalty=Settings.CONV_SPARSITY_PENALTY, linear_penalty=Settings.LINEAR_SPARSITY_PENALTY,
              latent_penalty=Settings.LATENT_SPARSITY_PENALTY):
    if max(conv_penalty, linear_penalty, latent_penalty) > 0:
        for m in model.modules():
            if isinstance(m, _BatchNorm):
                if isinstance(m, BatchNorm1d):
                    if m.weight.data.shape[0] == Settings.TOPOLOGY[0]:
                        m.weight.grad.data.add_(latent_penalty*torch.sign(m.weight.data))  # L1
                        # set weights below a threshold to 0
                        mask = torch.abs(m.weight.data) < 1e-4
                        m.weight.data[mask] = 0
                        m.weight.grad.data[mask] = 0
                    else:
                        m.weight.grad.data.add_(linear_penalty*torch.sign(m.weight.data))  # L1
                else:
                    m.weight.grad.data.add_(conv_penalty*torch.sign(m.weight.data))  # L1
