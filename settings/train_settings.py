from typing import Union, Type

from models.conv_ae import ConvAE
from models.ff_ae import FeedforwardAE
from settings.data_settings import CHANNEL_STACKING
from settings.global_settings import *

NETWORK: Union[Type[ConvAE], Type[FeedforwardAE]]

if ds == DatasetOption.CIFAR10:
    # CONV
    NETWORK = ConvAE
    L2REG = 0
    CONV_SPARSITY_PENALTY = 1e-4
    LINEAR_SPARSITY_PENALTY = CONV_SPARSITY_PENALTY
    LATENT_SPARSITY_PENALTY = CONV_SPARSITY_PENALTY
    BATCH_SIZE = 32
    LR = 2e-4 * (BATCH_SIZE ** 0.5)
    # latent_size, hidden_layers, multiplier
    # TOPOLOGY = [3, 4, 3]
    TOPOLOGY = [10, 4, 6, 3, 32]
    DRAW_EPOCHS = 20
elif ds == DatasetOption.FASHION_MNIST:
    # CONV
    NETWORK = ConvAE
    L2REG = 0
    CONV_SPARSITY_PENALTY = 1e-4
    LINEAR_SPARSITY_PENALTY = CONV_SPARSITY_PENALTY
    LATENT_SPARSITY_PENALTY = CONV_SPARSITY_PENALTY
    BATCH_SIZE = 64
    LR = 2e-4 * (BATCH_SIZE ** 0.5)
    # latent_size, hidden_layers, multiplier
    # TOPOLOGY = [3, 4, 3]
    TOPOLOGY = [3, 4, 1]
    DRAW_EPOCHS = 20
elif ds == DatasetOption.SYNTHETIC_FLAT:
    # FF
    NETWORK = FeedforwardAE
    L2REG = 0
    CONV_SPARSITY_PENALTY = 0
    LINEAR_SPARSITY_PENALTY = 1e-4
    LATENT_SPARSITY_PENALTY = 1e-4
    BATCH_SIZE = 64
    LR = 1e-3
    # latent_size, hidden_layers, multiplier
    TOPOLOGY = [8, 2, 1]
    DRAW_EPOCHS = 20
elif ds == DatasetOption.SYNTHETIC_IM:
    # CONV
    NETWORK = ConvAE
    L2REG = 0
    CONV_SPARSITY_PENALTY = 1e-4
    LINEAR_SPARSITY_PENALTY = 1e-3
    LATENT_SPARSITY_PENALTY = 20e-2
    BATCH_SIZE = 64
    LR = 1e-3 * (BATCH_SIZE ** 0.5)
    # latent_size, hidden_layers, multiplier
    TOPOLOGY = [20, 5, 6, CHANNEL_STACKING]
    DRAW_EPOCHS = 20
elif ds == DatasetOption.MNIST:
    # CONV
    NETWORK = ConvAE
    L2REG = 0
    CONV_SPARSITY_PENALTY = 1e-4
    LINEAR_SPARSITY_PENALTY = 1e-3
    LATENT_SPARSITY_PENALTY = 20e-2
    BATCH_SIZE = 64
    LR = 1e-3 * (BATCH_SIZE ** 0.5)
    # latent_size, hidden_layers, multiplier
    TOPOLOGY = [20, 5, 6]
    DRAW_EPOCHS = 30
else:
    raise ValueError("Unknown DatasetOption", ds)
