import os
import random
import torch
from torch import nn
from torchvision.transforms import ToTensor
from tqdm import tqdm
from torch.nn import MSELoss
from torch.optim import Adam
from torch.utils.data import DataLoader

from eval.eval import eval_network
from models.conv_ae import ConvAE
from torchvision.datasets import FashionMNIST

from procedures.channel_prune import find_channel_mask
from procedures.test import test
from procedures.train import train

LR = 1e-3
L2REG = 0
SPARSITY_PENALTY = 1e-4
BATCH_SIZE = 16
SETTINGS = [6, 4, 6]

# fractions of the network to keep
RATIOS = [0.75, 0.5, 0.25, 0.15, 0.1, 0.05]

unique_id = hex(random.randint(16**8, 16**9))[2:]
print(f"RUN ID: {unique_id}")
folder = f"runs/{unique_id}"

network = ConvAE(*SETTINGS)

train_set = FashionMNIST("data/", train=True, download=True, transform=ToTensor())
test_set = FashionMNIST("data/", train=False, download=True, transform=ToTensor())

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True)

optimiser = Adam(network.parameters(), lr=LR)
criterion = MSELoss()


if __name__ == '__main__':
    os.makedirs(folder, exist_ok=True)
    torch.save(network.state_dict(), folder + f"/starting_params-{SETTINGS}.pth")

    for epoch in range(15):
        # save the current mask drawn based on channel pruning
        for r in RATIOS:
            torch.save(list(find_channel_mask(network, r).values()), f"{folder}/keep-{r}-epoch-{epoch}.pth")

        train_loss = train(network, optimiser, criterion, train_loader)

        test_loss = test(network, criterion, test_loader)

        eval_network(network, test_set[0][0].view(1, 1, 28, 28))

        torch.save(network.state_dict(), "network.pth")
        print(train_loss, test_loss)
