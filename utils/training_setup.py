from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor

from settings.train_settings import BATCH_SIZE
from utils.file import change_working_dir


def get_loaders():
    change_working_dir()
    train_set = FashionMNIST("data/", train=True, download=True, transform=ToTensor())
    test_set = FashionMNIST("data/", train=False, download=True, transform=ToTensor())

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    return train_loader, test_loader
