import json
import os
from os.path import *

import torch
from torch import nn
from torch.nn import MSELoss
from torch.optim import Adam

from models.conv_ae import ConvAE
from procedures.test import test
from procedures.train import train
from settings.train_settings import LR, RETRAIN_EPOCHS
from utils.training_setup import get_loaders

run_id = "withbn"
ratio = 0.5
draw_epoch = 4
draw_sub_epoch = 4
mask = f"runs/{run_id}/keep-{ratio}-epoch-{draw_epoch}-{draw_sub_epoch}.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    train_loader, test_loader = get_loaders()

    params = [x for x in os.listdir(dirname(mask)) if x.startswith("starting_params-")][0]
    params = join(dirname(mask), params)
    settings = params.removesuffix(".pth").split("-")[-1].strip("[]")
    settings = [int(x) for x in settings.split(",")]

    unpruned = ConvAE(*settings)
    unpruned.load_state_dict(torch.load(params))

    pruned = ConvAE.init_from_checkpoint(run_id, ratio, draw_epoch, draw_sub_epoch)

    pruned_reset = ConvAE.init_from_checkpoint(run_id, ratio, draw_epoch, draw_sub_epoch)
    # reset weights
    for m in pruned_reset.modules():
        if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight.data)
            m.bias.data.zero_()

    networks = [
        ("unpruned", unpruned),
        ("pruned", pruned),
        ("pruned_reset", pruned_reset)
    ]

    graph_data = {}

    for net_tag, net in networks:
        print(f"Retraining: {net_tag}")
        current_graph_data = []
        graph_data[net_tag] = current_graph_data
        optimiser = Adam(net.parameters(), lr=LR)
        criterion = MSELoss()

        for epoch in range(RETRAIN_EPOCHS):
            train_loss = train(net, optimiser, criterion, train_loader, device)
            test_loss = test(net, criterion, test_loader, device)

            print(train_loss, test_loss)
            current_graph_data.append((epoch, train_loss, test_loss))

            with open(f"graphs/graph_data/reset-{run_id}-{ratio}.json", "w") as write_file:
                write_file.write(json.dumps(graph_data))
