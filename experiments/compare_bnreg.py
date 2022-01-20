import json
import os

import torch
from torch import nn
from torch.nn import MSELoss
from torch.optim import Adam

from experiments.progressive_mask_drawing import train_and_draw_tickets
from models.conv_ae import ConvAE
from procedures.test import test
from procedures.train import train
from settings.retrain_settings import *
from settings.train_settings import NETWORK, TOPOLOGY, SPARSITY_PENALTY
from utils.file import get_topology_of_run, get_params_of_run, change_working_dir
from datasets.get_loaders import get_loaders
from utils.get_run_id import all_runs_matching
from utils.misc import generate_random_str
from os.path import *


if __name__ == '__main__':
    change_working_dir()
    draw_epoch = 1
    draw_sub_epoch = 4
    ratio = 0.7
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # group_run_id = "BNREG_GROUP-" + generate_random_str()
    group_run_id = "BNREG_GROUP-8637645a6"

    base_folder = join("runs", group_run_id)
    if not isdir(base_folder):
        os.makedirs(base_folder)

        # get a single network initialisation and save it
        base_network = NETWORK(*TOPOLOGY).to(device)
        torch.save(base_network.state_dict(), join(base_folder, f"starting_params-{TOPOLOGY}.pth"))

        bn_regs = [0, 1e-5, 1e-4, 1e-3, 1e-2]

        # for each bn_reg, override the setting and run a normal mask drawing
        for bn_reg in bn_regs:
            SPARSITY_PENALTY = bn_reg
            net = NETWORK(*TOPOLOGY).to(device)
            net.init_from_checkpoint(group_run_id, None, None)
            train_and_draw_tickets(net, str(bn_reg), folder_root=base_folder)

    run_ids = [x for x in os.listdir(base_folder) if isdir(join(base_folder, x))]
    networks = []
    for run_id in run_ids:
        net = ConvAE.init_from_checkpoint(join(group_run_id, run_id), ratio, draw_epoch, draw_sub_epoch,
                                          param_epoch=RETRAIN_RESUME_EPOCH).to(device)
        networks.append((run_id, net))

    graph_data = {}
    train_loader, test_loader = get_loaders()

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

            with open(f"graphs/graph_data/{group_run_id}-{ratio}.json", "w") as write_file:
                write_file.write(json.dumps(graph_data))
