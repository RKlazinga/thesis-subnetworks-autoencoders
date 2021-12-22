from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor

from datasets.dataset_options import DatasetOption
from datasets.flat_generators import flat_gaussian, random_sine_gaussian
from datasets.im_generators import dummy_gen
from datasets.synthetic_flat import SyntheticFlat
from datasets.synthetic_im import Synthetic
from settings.train_settings import BATCH_SIZE
from settings.global_settings import ds
from utils.file import change_working_dir


def get_loaders(dataset=ds):
    change_working_dir()
    if dataset == DatasetOption.FASHION_MNIST:
        train_set = FashionMNIST("data/", train=True, download=True, transform=ToTensor())
        test_set = FashionMNIST("data/", train=False, download=True, transform=ToTensor())
    elif dataset == DatasetOption.SYNTHETIC_IM:
        train_set = Synthetic("train", dummy_gen, 100)
        test_set = Synthetic("test", dummy_gen, 10)
    elif dataset == DatasetOption.SYNTHETIC_FLAT:
        train_set = SyntheticFlat("train", random_sine_gaussian, num=10000, keep_in_ram=True)
        test_set = SyntheticFlat("test", random_sine_gaussian, num=1000, keep_in_ram=True)
    else:
        raise ValueError(f"Unknown dataset enum value: {dataset}")

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4 if dataset == DatasetOption.FASHION_MNIST else 0)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4 if dataset == DatasetOption.FASHION_MNIST else 0)

    return train_loader, test_loader
