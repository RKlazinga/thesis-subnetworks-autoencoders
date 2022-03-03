import os
import random
import shutil
from typing import Callable

from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from tqdm import tqdm

from utils.file import change_working_dir


class Synthetic(Dataset):
    def __init__(self, mode: str, generator: Callable[[], Image.Image], num: int, regen=False, keep_in_ram=True):
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
        self.keep_in_ram = keep_in_ram

        if self.keep_in_ram:
            self.imgs = [generator() for _ in tqdm(range(self.num), desc="Generating tensors")]
        else:
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
        if self.keep_in_ram:
            return self.imgs[index]
        return ToTensor()(Image.open(f"{self.imgs[index]}.png"))

    def __len__(self):
        return len(self.imgs)
