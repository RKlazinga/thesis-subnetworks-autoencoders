import json
import torch
from torch import nn
from torch.nn import MSELoss
from torch.optim import Adam

from models.conv_ae import ConvAE
from procedures.test import test
from procedures.train import train
from settings.retrain_settings import *
from utils.file import get_topology_of_run, get_params_of_run
from datasets.get_loaders import get_loaders
from utils.get_run_id import all_runs_matching

PREFIX = "bnreg"
draw_epoch = 4
draw_sub_epoch = 4
ratio = 0.7
run_ids = all_runs_matching(PREFIX)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    train_loader, test_loader = get_loaders()

    networks = []
    for run_id in run_ids:
        net = ConvAE.init_from_checkpoint(run_id, ratio, draw_epoch, draw_sub_epoch, param_epoch=RETRAIN_RESUME_EPOCH).to(device)
        tag = run_id.split("-[")[0].removeprefix(PREFIX)
        networks.append((tag, net))

    graph_data = {}

    for net_tag, net in networks:
        print(f"Retraining: {net_tag}")
        current_graph_data = []
        graph_data[net_tag] = current_graph_data
        optimiser = Adam(net.parameters(), lr=RETRAIN_LR)
        criterion = MSELoss()

        for epoch in range(RETRAIN_EPOCHS):
            train_loss = train(net, optimiser, criterion, train_loader, device)
            test_loss = test(net, criterion, test_loader, device)

            print(train_loss, test_loss)
            current_graph_data.append((epoch, train_loss, test_loss))

            with open(f"graphs/graph_data/bnreg-{ratio}.json", "w") as write_file:
                write_file.write(json.dumps(graph_data))
