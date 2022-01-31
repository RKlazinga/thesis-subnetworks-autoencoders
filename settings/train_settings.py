from models.conv_ae import ConvAE
from models.ff_ae import FeedforwardAE
from settings.global_settings import *

if ds == DatasetOption.FASHION_MNIST:
    # CONV
    NETWORK = ConvAE
    L2REG = 0  # 1e-4
    SPARSITY_PENALTY = 1e-4
    BATCH_SIZE = 32
    LR = 1e-4 * (BATCH_SIZE ** 0.5)
    TOPOLOGY = [6, 4, 6]
    DRAW_EPOCHS = 20
elif ds == DatasetOption.SYNTHETIC_FLAT:
    # FF
    NETWORK = FeedforwardAE
    L2REG = 0  # 1e-3
    SPARSITY_PENALTY = 1e-2
    BATCH_SIZE = 32
    LR = 1e-3 * (BATCH_SIZE ** 0.5)
    # latent_size, hidden_layers, multiplier
    TOPOLOGY = [4, 2, 1]
    DRAW_EPOCHS = 16
else:
    raise ValueError("Unknown DatasetOption", ds)
