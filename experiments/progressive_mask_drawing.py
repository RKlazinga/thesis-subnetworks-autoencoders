import os
import random
import torch
from torch.nn import MSELoss
from torch.optim import Adam

from models.conv_ae import ConvAE
from procedures.ticket_drawing.without_redist import find_channel_mask_no_redist
from procedures.test import test
from procedures.train import train
from settings.train_settings import *
from settings.prune_settings import *
from utils.ensure_correct_folder import change_working_dir
from utils.training_setup import get_loaders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
network = ConvAE(*TOPOLOGY).to(device)

change_working_dir()
train_loader, test_loader = get_loaders()

optimiser = Adam(network.parameters(), lr=LR)
criterion = MSELoss()

# TODO look at learning rate scheduling

if __name__ == '__main__':
    unique_id = hex(random.randint(16**8, 16**9))[2:]
    print(f"RUN ID: {unique_id}")
    folder = f"runs/{unique_id}"

    os.makedirs(folder, exist_ok=True)
    torch.save(network.state_dict(), folder + f"/starting_params-{TOPOLOGY}.pth")

    for epoch in range(DRAW_EPOCHS):
        # save the current mask drawn based on channel pruning

        def prune_snapshot(iter: int, epoch=epoch):
            for r in PRUNE_RATIOS:
                torch.save(list(find_channel_mask_no_redist(network, r).values()), f"{folder}/keep-{r}-epoch-{epoch}-{iter}.pth")

        if epoch == 0:
            prune_snapshot(0)

        train_loss = train(network, optimiser, criterion, train_loader, device, prune_snapshot_method=prune_snapshot)
        test_loss = test(network, criterion, test_loader, device)

        print(train_loss, test_loss)
        torch.save(network.state_dict(), folder + f"/trained-{epoch}.pth")
