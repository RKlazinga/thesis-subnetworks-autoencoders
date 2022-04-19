from enum import Enum, auto


class DatasetOption(Enum):
    FASHION_MNIST = auto()
    CIFAR10 = auto()
    SYNTHETIC_IM = auto()
    SYNTHETIC_FLAT = auto()
    MNIST = auto()
