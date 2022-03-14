# channel sparsity regularisation as introduced by Liu et al. 2017
# taken from https://github.com/RICE-EIC/Early-Bird-Tickets/blob/4a16ae0731edf288c48e000c1c2a51dc0433f4ef/main.py#L277
# "additional subgradient descent on the sparsity-induced penalty term"
import torch
from torch.nn.modules.batchnorm import _BatchNorm, BatchNorm1d
from settings.train_settings import SPARSITY_PENALTY, LATENT_SPARSITY_PENALTY, TOPOLOGY


def update_bn(model, sparsity_penalty=SPARSITY_PENALTY, latent_penalty=LATENT_SPARSITY_PENALTY):
    if sparsity_penalty > 0:
        for m in model.modules():
            if isinstance(m, _BatchNorm):
                if isinstance(m, BatchNorm1d) and m.weight.data.shape[0] == TOPOLOGY[0]:
                    # set weights below a threshold to 0
                    mask = m.weight.data < 1e-6

                    m.weight.grad.data.add_(latent_penalty*torch.sign(m.weight.data))  # L1
                    m.weight.data[mask] = 0
                    m.weight.grad.data[mask] = 0
                else:
                    m.weight.grad.data.add_(sparsity_penalty*torch.sign(m.weight.data))  # L1
