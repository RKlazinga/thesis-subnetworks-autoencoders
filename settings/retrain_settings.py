from settings.global_settings import *
from settings.train_settings import BATCH_SIZE

if ds == DatasetOption.FASHION_MNIST:
    # CONV
    RETRAIN_LR = 2e-4 * (BATCH_SIZE ** 0.5)
    RETRAIN_L2REG = 1e-4
    RETRAIN_RESUME_EPOCH = 1
    RETRAIN_EPOCHS = 30
elif ds == DatasetOption.SYNTHETIC_FLAT:
    # FF
    RETRAIN_LR = 1e-3 * (BATCH_SIZE ** 0.5)
    RETRAIN_L2REG = 1e-4
    RETRAIN_RESUME_EPOCH = 1
    RETRAIN_EPOCHS = 15
else:
    raise ValueError("Unknown DatasetOption", ds)
