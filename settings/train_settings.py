from models.ff_ae import FeedforwardAE

# CONV
# NETWORK = ConvAE
# L2REG = 1e-4
# SPARSITY_PENALTY = 1e-4
# BATCH_SIZE = 32
# LR = 1e-4 * (BATCH_SIZE ** 0.5)
# TOPOLOGY = [6, 4, 6]
# DRAW_EPOCHS = 20


# FF
NETWORK = FeedforwardAE
L2REG = 1e-4
SPARSITY_PENALTY = 1e-4
BATCH_SIZE = 32
LR = 1e-3 * (BATCH_SIZE ** 0.5)
# latent_size, hidden_layers, multiplier
TOPOLOGY = [2, 2, 1]
DRAW_EPOCHS = 30
