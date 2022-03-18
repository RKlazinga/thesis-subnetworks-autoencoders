from typing import Union, Type

from models.conv_ae import ConvAE
from models.ff_ae import FeedforwardAE
from settings.global_settings import *

NETWORK: Union[Type[ConvAE], Type[FeedforwardAE]]

if ds == DatasetOption.FASHION_MNIST:
    # CONV
    NETWORK = ConvAE
    L2REG = 0
    SPARSITY_PENALTY = 1e-4
    LATENT_SPARSITY_PENALTY = SPARSITY_PENALTY
    BATCH_SIZE = 64
    LR = 2e-4 * (BATCH_SIZE ** 0.5)
    # latent_size, hidden_layers, multiplier
    # TOPOLOGY = [3, 4, 3]
    TOPOLOGY = [6, 4, 6]
    DRAW_EPOCHS = 20
elif ds == DatasetOption.SYNTHETIC_FLAT:
    # FF
    NETWORK = FeedforwardAE
    L2REG = 0
    SPARSITY_PENALTY = 5e-2
    LATENT_SPARSITY_PENALTY = SPARSITY_PENALTY
    BATCH_SIZE = 64
    LR = 2e-4 * (BATCH_SIZE ** 0.5)
    # latent_size, hidden_layers, multiplier
    TOPOLOGY = [8, 2, 1]
    DRAW_EPOCHS = 20
elif ds == DatasetOption.SYNTHETIC_IM:
    # CONV
    NETWORK = ConvAE
    L2REG = 0
    SPARSITY_PENALTY = 2e-4
    LATENT_SPARSITY_PENALTY = 10e-2  # SPARSITY_PENALTY
    BATCH_SIZE = 64
    LR = 2e-4 * (BATCH_SIZE ** 0.5)
    # latent_size, hidden_layers, multiplier
    TOPOLOGY = [12, 5, 6]
    DRAW_EPOCHS = 30
elif ds == DatasetOption.MNIST:
    # CONV
    NETWORK = ConvAE
    L2REG = 0
    SPARSITY_PENALTY = 2e-4
    LATENT_SPARSITY_PENALTY = 8e-2  # SPARSITY_PENALTY
    BATCH_SIZE = 64
    LR = 4e-4 * (BATCH_SIZE ** 0.5)
    # latent_size, hidden_layers, multiplier
    TOPOLOGY = [20, 5, 6]
    DRAW_EPOCHS = 30
else:
    raise ValueError("Unknown DatasetOption", ds)
