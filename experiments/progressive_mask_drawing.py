import os
import random
import torch
from torch.nn import MSELoss
from torch.optim import Adam

from models.conv_ae import ConvAE
from procedures.ticket_drawing.with_redist import find_channel_mask_redist
from procedures.ticket_drawing.without_redist import find_channel_mask_no_redist
from procedures.test import test
from procedures.train import train
from settings.train_settings import *
from settings.prune_settings import *
from utils.file import change_working_dir
from datasets.get_loaders import get_loaders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
network = ConvAE(*TOPOLOGY).to(device)

change_working_dir()
train_loader, test_loader = get_loaders()

optimiser = Adam(network.parameters(), lr=LR, weight_decay=L2REG)
criterion = MSELoss()

# TODO look at learning rate scheduling

if __name__ == '__main__':
    unique_id = hex(random.randint(16**8, 16**9))[2:]
    if PRUNE_WITH_REDIST:
        unique_id = "prop_redist-" + unique_id

    unique_id = f"{TOPOLOGY}-" + unique_id

    channel_mask_func = find_channel_mask_redist if PRUNE_WITH_REDIST else find_channel_mask_no_redist

    print(f"RUN ID: {unique_id}")
    folder = f"runs/{unique_id}"
    os.makedirs(folder)
    torch.save(network.state_dict(), folder + f"/starting_params-{TOPOLOGY}.pth")

    for epoch in range(1, DRAW_EPOCHS + 1):
        def prune_snapshot(iter: int, epoch=epoch):
            for r in PRUNE_RATIOS:
                torch.save(list(channel_mask_func(network, r).values()), f"{folder}/keep-{r}-epoch-{epoch}-{iter}.pth")

        train_loss = train(network, optimiser, criterion, train_loader, device, prune_snapshot_method=prune_snapshot)
        test_loss = test(network, criterion, test_loader, device)

        print(f"{epoch}/{DRAW_EPOCHS}: {round(train_loss, 8)} & {round(test_loss, 8)}")
        torch.save(network.state_dict(), folder + f"/trained-{epoch}.pth")
