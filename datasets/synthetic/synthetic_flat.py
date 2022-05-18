import os
import random
import shutil
from typing import Callable

import torch
from torch.utils.data import Dataset

from settings import Settings
from utils.file import change_working_dir


class SyntheticFlat(Dataset):
    def __init__(self, mode: str, generator: Callable[[], torch.Tensor], num: int, regen=False, keep_in_ram=True):
        """
        Load or generate a set of synthetic images.

        :param mode: Folder name to separately store splits. Usually "train" or "test".
        :param generator: Function that produces a random image, possibly based on a seed
        :param num: Size of the dataset in number of images.
                    Will only generate more images if the existing number is insufficient.
        :param regen: Whether to completely regenerate the cached synthetic images
        """
        super().__init__()
        self.mode = mode
        self.generator = generator
        self.num = num
        self.keep_in_ram = keep_in_ram

        change_working_dir()
        self.folder = f"data/flat-{mode}"

        if Settings.OVERRIDE_SYNTH_GENERATOR:
            self.keep_in_ram = True
            # generate data with the overridden generator function
            data = torch.Tensor(Settings.OVERRIDE_SYNTH_GENERATOR(self.num))
            data = data - torch.mean(data, dim=0)
            data = data / (torch.max(torch.abs(data), dim=0).values + 1e-6)
            self.imgs = [data[i] for i in range(data.shape[0])]
        else:
            if self.keep_in_ram:
                print(f"Generating {self.num} tensors")
                self.imgs = [generator() for _ in range(self.num)]
            else:
                if regen and os.path.isdir(self.folder):
                    shutil.rmtree(self.folder)

                os.makedirs(self.folder, exist_ok=True)

                existing_imgs = len(os.listdir(self.folder))

                if self.num > existing_imgs:
                    for i in range(existing_imgs, self.num):
                        t = generator()
                        torch.save(t, os.path.join(self.folder, f"{i}.pth"))

                existing_imgs = max(self.num, existing_imgs)

                self.imgs = random.sample(range(existing_imgs), self.num)
                self.imgs = [os.path.join(self.folder, f"{x}.pth") for x in self.imgs]

    def __getitem__(self, index):
        if self.keep_in_ram:
            return self.imgs[index]
        return torch.load(self.imgs[index])

    def __len__(self):
        return len(self.imgs)
