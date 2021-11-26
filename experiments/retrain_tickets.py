import os

import torch
from torch.nn import MSELoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor

from eval.eval import eval_network
from models.conv_ae import ConvAE
from procedures.test import test
from procedures.train import train
from settings.prune_settings import DRAW_PER_EPOCH

from settings.train_settings import *

run_id = "lre3"
ratio = 0.5
train_every = 4
checkpoint_folder = f"runs/{run_id}/"
graph_data_folder = f"graphs/graph_data/{run_id}"

train_set = FashionMNIST("data/", train=True, download=True, transform=ToTensor())
test_set = FashionMNIST("data/", train=False, download=True, transform=ToTensor())

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True)

criterion = MSELoss()


if __name__ == '__main__':
    masks = [x for x in os.listdir(checkpoint_folder) if x.startswith(f"keep-{ratio}-")]
    for draw_epoch in range(6):
        for sub_epoch in range(1, DRAW_PER_EPOCH + 1):
            if sub_epoch % train_every == 0:
                print(f"Retraining: ratio {ratio}, epoch {draw_epoch}, sub-epoch {sub_epoch}")
                network = ConvAE.init_from_checkpoint(run_id, ratio, draw_epoch, sub_epoch)
                optimiser = Adam(network.parameters(), lr=LR)

                for epoch in range(10):
                    train_loss = train(network, optimiser, criterion, train_loader)
                    test_loss = test(network, criterion, test_loader)

                    print(train_loss, test_loss)
