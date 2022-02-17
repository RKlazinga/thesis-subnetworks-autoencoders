import os
import random
import torch
from torch.nn import MSELoss
from torch.optim import Adam

from procedures.ticket_drawing.with_redist import find_channel_mask_redist
from procedures.ticket_drawing.without_redist import find_channel_mask_no_redist
from procedures.test import test
from procedures.train import train
from settings.train_settings import *
from settings.prune_settings import *
from utils.file import change_working_dir
from datasets.get_loaders import get_loaders
from utils.misc import generate_random_str, get_device


def train_and_draw_tickets(net, uid, folder_root="runs"):
    channel_mask_func = find_channel_mask_redist if PRUNE_WITH_REDIST else find_channel_mask_no_redist
    device = get_device()

    train_loader, test_loader = get_loaders()
    optimiser = Adam(net.parameters(), lr=LR, weight_decay=L2REG)
    criterion = MSELoss()

    print(f"RUN ID: {uid}")
    folder = f"{folder_root}/{uid}"
    os.makedirs(folder)
    torch.save(net.state_dict(), folder + f"/starting_params-{TOPOLOGY}.pth")

    for epoch in range(1, DRAW_EPOCHS + 1):
        def prune_snapshot(iteration: int, epoch=epoch):
            for r in PRUNE_RATIOS:
                torch.save(list(channel_mask_func(net, r).values()), f"{folder}/prune-{r}-epoch-{epoch}-{iteration}.pth")

        train_loss = train(net, optimiser, criterion, train_loader, device, prune_snapshot_method=prune_snapshot)
        test_loss = test(net, criterion, test_loader, device)

        print(f"{epoch}/{DRAW_EPOCHS}: {round(train_loss, 8)} & {round(test_loss, 8)}")
        torch.save(net.state_dict(), folder + f"/trained-{epoch}.pth")

# TODO look at learning rate scheduling


if __name__ == '__main__':
    unique_id = generate_random_str()
    device = get_device()
    _network = NETWORK(*TOPOLOGY).to(device)

    change_working_dir()

    if PRUNE_WITH_REDIST:
        unique_id = "prop_redist-" + unique_id

    unique_id = f"{TOPOLOGY}-" + unique_id
    train_and_draw_tickets(_network, unique_id)
