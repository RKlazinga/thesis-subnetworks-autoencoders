import os
import pickle

from datasets.dataset_options import DatasetOption


class Settings:
    # GLOBAL
    DS = DatasetOption.SYNTHETIC_FLAT
    print(f"DatasetOption: {DS.name}")
    RUN_FOLDER = "runs"

    # DATA
    NORMAL_STD_DEV = 0.01
    FLAT_DATAPOINTS = 16
    NUM_VARIABLES = 4
    CHANNEL_STACKING = 1

    OVERRIDE_SYNTH_GENERATOR = None

    TRAIN_SIZE = 20000
    TEST_SIZE = TRAIN_SIZE // 10

    # PRUNE
    DRAW_PER_EPOCH = 4
    PRUNE_RATIOS = [0.1, 0.3, 0.5, 0.7]
    PRUNE_LIMIT = 0.8
    PRUNE_WITH_REDIST = False

    REG_SCALING_TYPE = "off"
    MAX_LOSS = 0.1
    MIN_LOSS = 0
    REG_MULTIPLIER = 1

    # TRAIN
    if DS == DatasetOption.FASHION_MNIST:
        # CONV
        L2REG = 0
        CONV_SPARSITY_PENALTY = 1e-4
        LINEAR_SPARSITY_PENALTY = CONV_SPARSITY_PENALTY
        LATENT_SPARSITY_PENALTY = 0
        BATCH_SIZE = 64
        LR = 2e-4 * (BATCH_SIZE ** 0.5)
        # latent_size, hidden_layers, multiplier
        TOPOLOGY = [3, 5, 3]
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
        TOPOLOGY = [8, 4, 1]
        DRAW_EPOCHS = 20
    elif DS == DatasetOption.SYNTHETIC_IM:
        # CONV
        L2REG = 0
        CONV_SPARSITY_PENALTY = 1e-4
        LINEAR_SPARSITY_PENALTY = 1e-3
        LATENT_SPARSITY_PENALTY = 1e-3
        BATCH_SIZE = 64
        LR = 1e-2
        # latent_size, conv_layers, multiplier
        TOPOLOGY = [8, 5, 12, CHANNEL_STACKING]
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
        RETRAIN_L2REG = 0
        RETRAIN_RESUME_EPOCH = 1
        RETRAIN_EPOCHS = 20
    elif DS == DatasetOption.SYNTHETIC_FLAT:
        # FF
        RETRAIN_LR = 1e-3 * (BATCH_SIZE ** 0.5)
        RETRAIN_L2REG = 0
        RETRAIN_RESUME_EPOCH = 1
        RETRAIN_EPOCHS = 15

    @classmethod
    def serialise(cls):
        state = {var: getattr(cls, var) for var in dir(cls) if not var.startswith("_")
                 and var.isupper() and var not in ["NETWORK", "OVERRIDE_SYNTH_GENERATOR", "_DEFAULT"]}
        return pickle.dumps(state)

    @classmethod
    def deserialise(cls, state: bytes):
        state = pickle.loads(state)
        for var, value in state.items():
            if var != "_DEFAULT":
                setattr(cls, var, value)

    @classmethod
    def reset(cls):
        if hasattr(cls, "_DEFAULT"):
            cls.deserialise(cls._DEFAULT)

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
        with open(os.path.join(folder, "settings.bin"), "wb") as writefile:
            writefile.write(cls.serialise())

    @classmethod
    def from_disk(cls, folder):
        with open(os.path.join(folder, "settings.bin"), "rb") as readfile:
            cls.deserialise(readfile.read())

    @classmethod
    def compare_to(cls, folder):
        own_state = pickle.loads(cls.serialise())
        # load folder state
        with open(os.path.join(folder, "settings.bin"), "rb") as readfile:
            folder_state = pickle.loads(readfile.read())
        sentinel = object()

        for key in set(folder_state.keys()).union(set(own_state.keys())):
            f_v = sentinel
            own_v = sentinel

            if key in folder_state:
                f_v = folder_state[key]
            else:
                print(f"Old state did not contain key {key}")
            if key in own_state:
                own_v = own_state[key]
            else:
                print(f"New state is missing key {key}")

            if f_v != sentinel and own_v != sentinel:
                if f_v != own_v:
                    print(f"Value mismatch {key}: old={f_v} new={own_v}")


if not hasattr(Settings, "_DEFAULT"):
    Settings._DEFAULT = Settings.serialise()
