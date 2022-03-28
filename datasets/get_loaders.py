from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST, MNIST
from torchvision.transforms import ToTensor

from datasets.dataset_options import DatasetOption
from datasets.synthetic.flat_generators import random_sine_gaussian
from datasets.synthetic.im_generators import sine_2d, stacked_sine2d
from datasets.synthetic.synthetic_flat import SyntheticFlat
from datasets.synthetic.synthetic_im import Synthetic
from settings.data_settings import TRAIN_SIZE, TEST_SIZE
from settings.train_settings import BATCH_SIZE
from settings.global_settings import ds
from utils.file import change_working_dir


def get_loaders(dataset=ds):
    change_working_dir()
    if dataset == DatasetOption.FASHION_MNIST:
        train_set = FashionMNIST("data/", train=True, download=True, transform=ToTensor())
        test_set = FashionMNIST("data/", train=False, download=True, transform=ToTensor())
    elif dataset == DatasetOption.SYNTHETIC_IM:
        train_set = Synthetic("train", stacked_sine2d, num=TRAIN_SIZE, keep_in_ram=True)
        test_set = Synthetic("test", stacked_sine2d, num=TEST_SIZE, keep_in_ram=True)
    elif dataset == DatasetOption.SYNTHETIC_FLAT:
        train_set = SyntheticFlat("train", random_sine_gaussian, num=TRAIN_SIZE, keep_in_ram=True)
        test_set = SyntheticFlat("test", random_sine_gaussian, num=TEST_SIZE, keep_in_ram=True)
    elif dataset == DatasetOption.MNIST:
        train_set = MNIST("data/", train=True, download=True, transform=ToTensor())
        test_set = MNIST("data/", train=False, download=True, transform=ToTensor())
    else:
        raise ValueError(f"Unknown dataset enum value: {dataset}")

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=2 if dataset == DatasetOption.FASHION_MNIST else 0)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True,
                             num_workers=0 if dataset == DatasetOption.FASHION_MNIST else 0)

    return train_loader, test_loader
