# channel sparsity regularisation as introduced by Liu et al. 2017
# taken from https://github.com/RICE-EIC/Early-Bird-Tickets/blob/4a16ae0731edf288c48e000c1c2a51dc0433f4ef/main.py#L277
# "additional subgradient descent on the sparsity-induced penalty term"
import torch
from torch import nn


def updateBN(model, sparsity_penalty=1e-4):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.weight.grad.data.add_(sparsity_penalty*torch.sign(m.weight.data))  # L1
