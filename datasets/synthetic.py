import os
import random
import shutil
from typing import Callable, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

from datasets.synth_generators import dummy_gen
from utils.file import change_working_dir


class Synthetic(Dataset):
    def __init__(self, mode: str, generator: Callable[[], Image.Image], num: int, regen=False):
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
        change_working_dir()
        self.folder = f"data/{mode}"

        if regen and os.path.isdir(self.folder):
            shutil.rmtree(self.folder)

        os.makedirs(self.folder, exist_ok=True)

        existing_imgs = len(os.listdir(self.folder))

        if self.num > existing_imgs:
            for i in range(existing_imgs, self.num):
                im = generator()
                im.save(os.path.join(self.folder, f"{i}.png"))

        existing_imgs = max(self.num, existing_imgs)

        self.imgs = random.sample(range(existing_imgs), self.num)

    def __getitem__(self, index):
        return ToTensor()(Image.open(f"{self.imgs[index]}.png"))

if __name__ == '__main__':
    Synthetic("trial", dummy_gen, 10)