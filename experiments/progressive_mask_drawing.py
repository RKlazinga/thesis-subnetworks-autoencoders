import os
import random
import torch
from torchvision.transforms import ToTensor
from torchvision.datasets import FashionMNIST
from torch.nn import MSELoss
from torch.optim import Adam
from torch.utils.data import DataLoader

from eval.eval import eval_network
from models.conv_ae import ConvAE
from procedures.draw_channel_pruning import find_channel_mask
from procedures.test import test
from procedures.train import train
from settings.train_settings import *
from settings.prune_settings import *

unique_id = hex(random.randint(16**8, 16**9))[2:]
print(f"RUN ID: {unique_id}")
folder = f"runs/{unique_id}"

network = ConvAE(*TOPOLOGY)

train_set = FashionMNIST("data/", train=True, download=True, transform=ToTensor())
test_set = FashionMNIST("data/", train=False, download=True, transform=ToTensor())

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True)

optimiser = Adam(network.parameters(), lr=LR)
criterion = MSELoss()


if __name__ == '__main__':
    os.makedirs(folder, exist_ok=True)
    torch.save(network.state_dict(), folder + f"/starting_params-{TOPOLOGY}.pth")

    for epoch in range(10):
        # save the current mask drawn based on channel pruning

        def prune_snapshot(iter: int, epoch=epoch):
            for r in PRUNE_RATIOS:
                torch.save(list(find_channel_mask(network, r).values()), f"{folder}/keep-{r}-epoch-{epoch}-{iter}.pth")

        if epoch == 0:
            prune_snapshot(0)

        train_loss = train(network, optimiser, criterion, train_loader, prune_snapshot_method=prune_snapshot)
        test_loss = test(network, criterion, test_loader)

        eval_network(network, test_set[0][0].view(1, 1, 28, 28))

        print(train_loss, test_loss)
