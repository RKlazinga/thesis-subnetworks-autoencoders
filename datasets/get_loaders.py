from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor

from datasets.dataset_options import DatasetOption
from datasets.synth_generators import dummy_gen
from datasets.synthetic import Synthetic
from settings.train_settings import BATCH_SIZE
from utils.file import change_working_dir


def get_loaders(dataset: DatasetOption = DatasetOption.FASHION_MNIST):
    change_working_dir()
    if dataset == DatasetOption.FASHION_MNIST:
        train_set = FashionMNIST("data/", train=True, download=True, transform=ToTensor())
        test_set = FashionMNIST("data/", train=False, download=True, transform=ToTensor())
    elif dataset == DatasetOption.SYNTHETIC:
        train_set = Synthetic("train", dummy_gen, 100)
        test_set = Synthetic("test", dummy_gen, 10)
    else:
        raise ValueError(f"Unknown dataset enum value: {dataset}")

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    return train_loader, test_loader
