import json
import os

import torch
from torch.nn import MSELoss
from torch.nn.modules.batchnorm import _BatchNorm
from torch.optim import Adam

from datasets.dataset_options import DatasetOption
from procedures.ticket_drawing import find_channel_mask
from procedures.test import test
from procedures.train import train
from settings.s import Settings
from utils.file import change_working_dir, get_all_current_settings
from datasets.get_loaders import get_loaders
from utils.misc import generate_random_str, dev


def train_and_draw_tickets(net, uid):
    train_loader, test_loader = get_loaders()

    # TODO consider split learning rate
    optimiser = Adam(net.parameters(), lr=Settings.LR)
    criterion = MSELoss()

    print(f"RUN ID: {uid}")
    folder = f"{Settings.RUN_FOLDER}/{uid}"
    os.makedirs(folder)
    os.makedirs(f"{folder}/masks")

    # save initial parameters
    torch.save(net.state_dict(), f"{folder}/starting_params-{Settings.TOPOLOGY}")

    # save all settings
    with open(f"{folder}/settings.md", "w") as writefile:
        writefile.write(get_all_current_settings())

    loss_graph_data = []
    loss_file = f"{folder}/loss_graph.json"

    for epoch in range(1, Settings.DRAW_EPOCHS + 1):
        def prune_snapshot(iteration: int, epoch=epoch):
            for r in Settings.PRUNE_RATIOS:
                torch.save(list(find_channel_mask(net, r).values()), f"{folder}/masks/prune-{r}-epoch-{epoch}-{iteration}.pth")

        train_loss = train(net, optimiser, criterion, train_loader, prune_snapshot_method=prune_snapshot)

        # disable running statistics
        for m in net.modules():
            if isinstance(m, _BatchNorm):
                m.track_running_stats = False
        test_loss = test(net, criterion, test_loader)

        loss_graph_data.append((epoch, train_loss, test_loss))

        with open(loss_file, "w") as write_file:
            write_file.write(json.dumps(loss_graph_data))

        # eval_network(net, next(iter(test_loader)), device)
        for m in net.modules():
            if isinstance(m, _BatchNorm):
                m.track_running_stats = True

        print(f"{epoch}/{Settings.DRAW_EPOCHS}: {round(train_loss, 8)} & {round(test_loss, 8)}")
        torch.save(net.state_dict(), folder + f"/trained-{epoch}.pth")


def main(prefix=None):
    unique_id = generate_random_str()

    change_working_dir()

    unique_id = f"{Settings.TOPOLOGY}-" + unique_id
    if Settings.DS == DatasetOption.SYNTHETIC_FLAT:
        unique_id = f"flat{Settings.NUM_VARIABLES}-" + unique_id
    if Settings.DS == DatasetOption.SYNTHETIC_IM:
        unique_id = f"clean_synthim{Settings.NUM_VARIABLES}-" + unique_id
    if Settings.DS == DatasetOption.MNIST:
        unique_id = "mnist-" + unique_id
    if Settings.DS == DatasetOption.CIFAR10:
        unique_id = "cifar-" + unique_id

    if prefix is not None:
        unique_id = prefix + unique_id

    _network = Settings.NETWORK(*Settings.TOPOLOGY).to(dev())

    train_and_draw_tickets(_network, unique_id)


if __name__ == '__main__':
    main()
