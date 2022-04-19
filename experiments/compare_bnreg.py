import os

import torch

from experiments.progressive_mask_drawing import train_and_draw_tickets
from models.conv_ae import ConvAE
from procedures.retrain import retrain_tagged_networks
from settings.s import Settings
from utils.file import change_working_dir
from utils.misc import generate_random_str, get_device
from os.path import *


if __name__ == '__main__':
    change_working_dir()
    draw_epoch = 8
    draw_sub_epoch = 4
    ratio = 0.3
    device = get_device()
    group_run_id = "BNREG_GROUP-" + generate_random_str()

    base_folder = join(Settings.RUN_FOLDER, group_run_id)
    if not isdir(base_folder):
        os.makedirs(base_folder)

        # get a single network initialisation and save it
        base_network = Settings.NETWORK(*Settings.TOPOLOGY).to(device)
        torch.save(base_network.state_dict(), join(base_folder, f"starting_params-{Settings.TOPOLOGY}.pth"))

        bn_regs = [0, 1e-5, 1e-4, 1e-3, 1e-2]

        # for each bn_reg, override the setting and run a normal mask drawing
        for bn_reg in bn_regs:
            SPARSITY_PENALTY = bn_reg
            net = Settings.NETWORK(*Settings.TOPOLOGY).to(device)
            net.init_from_checkpoint(group_run_id, None, None)
            train_and_draw_tickets(net, str(bn_reg), folder_root=base_folder)

    run_ids = [x for x in os.listdir(base_folder) if isdir(join(base_folder, x))]
    networks = []
    for run_id in run_ids:
        net = ConvAE.init_from_checkpoint(join(group_run_id, run_id), ratio, draw_epoch, draw_sub_epoch,
                                          param_epoch=Settings.RETRAIN_RESUME_EPOCH).to(device)
        networks.append((run_id, net))

    retrain_tagged_networks(networks, f"graph_data/group_retraining/{group_run_id}-{ratio}.json")
