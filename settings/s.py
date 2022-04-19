import os
import pickle

from datasets.dataset_options import DatasetOption


class Settings:
    # GLOBAL
    DS = DatasetOption.SYNTHETIC_FLAT
    RUN_FOLDER = "runs"

    # DATA
    NORMAL_STD_DEV = 0.01
    FLAT_DATAPOINTS = 16
    NUM_VARIABLES = 2
    CHANNEL_STACKING = 1  # multiplier on the number of input channels, to linearly increase the required latent count

    TRAIN_SIZE = 20000
    TEST_SIZE = TRAIN_SIZE // 10

    # PRUNE
    DRAW_PER_EPOCH = 4
    PRUNE_RATIOS = [0.1, 0.3, 0.5, 0.7]
    PRUNE_LIMIT = 0.8
    PRUNE_WITH_REDIST = False

    # TRAIN
    # NETWORK:
    if DS == DatasetOption.CIFAR10:
        # CONV
        L2REG = 0
        CONV_SPARSITY_PENALTY = 1e-4
        LINEAR_SPARSITY_PENALTY = CONV_SPARSITY_PENALTY
        LATENT_SPARSITY_PENALTY = CONV_SPARSITY_PENALTY
        BATCH_SIZE = 32
        LR = 2e-4 * (BATCH_SIZE ** 0.5)
        # latent_size, hidden_layers, multiplier
        TOPOLOGY = [10, 4, 6, 3, 32]
        DRAW_EPOCHS = 20
    elif DS == DatasetOption.FASHION_MNIST:
        # CONV
        L2REG = 0
        CONV_SPARSITY_PENALTY = 1e-4
        LINEAR_SPARSITY_PENALTY = CONV_SPARSITY_PENALTY
        LATENT_SPARSITY_PENALTY = CONV_SPARSITY_PENALTY
        BATCH_SIZE = 64
        LR = 2e-4 * (BATCH_SIZE ** 0.5)
        # latent_size, hidden_layers, multiplier
        TOPOLOGY = [3, 4, 1]
        DRAW_EPOCHS = 20
    elif DS == DatasetOption.SYNTHETIC_FLAT:
        # FF
        L2REG = 0
        CONV_SPARSITY_PENALTY = 0
        LINEAR_SPARSITY_PENALTY = 1e-4
        LATENT_SPARSITY_PENALTY = 1e-4
        BATCH_SIZE = 64
        LR = 1e-3
        # latent_size, hidden_layers, multiplier
        TOPOLOGY = [8, 2, 1]
        DRAW_EPOCHS = 20
    elif DS == DatasetOption.SYNTHETIC_IM:
        # CONV
        L2REG = 0
        CONV_SPARSITY_PENALTY = 1e-4
        LINEAR_SPARSITY_PENALTY = 1e-3
        LATENT_SPARSITY_PENALTY = 20e-2
        BATCH_SIZE = 64
        LR = 1e-3 * (BATCH_SIZE ** 0.5)
        # latent_size, hidden_layers, multiplier
        TOPOLOGY = [20, 5, 6, CHANNEL_STACKING]
        DRAW_EPOCHS = 20
    elif DS == DatasetOption.MNIST:
        # CONV
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
        raise ValueError("Unknown DatasetOption for train settings", DS)

    # RETRAIN
    if DS == DatasetOption.FASHION_MNIST:
        # CONV
        RETRAIN_LR = 2e-4 * (BATCH_SIZE ** 0.5)
        RETRAIN_L2REG = 0  # 1e-4
        RETRAIN_RESUME_EPOCH = 1
        RETRAIN_EPOCHS = 20  # 30
    elif DS == DatasetOption.SYNTHETIC_FLAT:
        # FF
        RETRAIN_LR = 1e-3 * (BATCH_SIZE ** 0.5)
        RETRAIN_L2REG = 0  # 1e-4
        RETRAIN_RESUME_EPOCH = 1
        RETRAIN_EPOCHS = 15
    else:
        raise ValueError("Unknown DatasetOption for retrain settings", DS)

    @classmethod
    @property
    def NETWORK(cls):
        from models.conv_ae import ConvAE
        from models.ff_ae import FeedforwardAE
        if cls.DS == DatasetOption.SYNTHETIC_FLAT:
            return FeedforwardAE
        return ConvAE

    @classmethod
    def to_disk(cls, folder):
        state = {var: getattr(cls, var) for var in dir(cls) if not var.startswith("__") and var.isupper()}
        os.path.join(folder, "settings.json")
        print(pickle.dumps(state))

    @classmethod
    def from_disk(cls):
        pass


if __name__ == '__main__':
    print(Settings.to_disk(""))
