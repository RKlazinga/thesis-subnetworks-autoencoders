import json
import torch
from torch import nn
from torch.nn import MSELoss
from torch.optim import Adam

from models.conv_ae import ConvAE
from procedures.test import test
from procedures.train import train
from settings.retrain_settings import RETRAIN_EPOCHS, RETRAIN_LR
from utils.file import get_topology_of_run, get_params_of_run
from datasets.get_loaders import get_loaders

run_id = "[6, 4, 6]-425222fd6"
ratio = 0.5
draw_epoch = 9
draw_sub_epoch = 4
mask = f"runs/{run_id}/keep-{ratio}-epoch-{draw_epoch}-{draw_sub_epoch}.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    train_loader, test_loader = get_loaders()

    unpruned = ConvAE(*get_topology_of_run(run_id)).to(device)
    unpruned.load_state_dict(get_params_of_run(run_id, device=device))

    pruned = ConvAE.init_from_checkpoint(run_id, ratio, draw_epoch, draw_sub_epoch).to(device)

    pruned_1 = ConvAE.init_from_checkpoint(run_id, ratio, draw_epoch, draw_sub_epoch, param_epoch=1).to(device)

    pruned_2 = ConvAE.init_from_checkpoint(run_id, ratio, draw_epoch, draw_sub_epoch, param_epoch=2).to(device)

    pruned_continue = ConvAE.init_from_checkpoint(run_id, ratio, draw_epoch, draw_sub_epoch, param_epoch=9).to(device)

    pruned_reset = ConvAE.init_from_checkpoint(run_id, ratio, draw_epoch, draw_sub_epoch).to(device)
    # reset weights
    for m in pruned_reset.modules():
        if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight.data)
            m.bias.data.zero_()

    networks = [
        ("unpruned", unpruned),
        ("pruned_no_resume", pruned),
        ("pruned_resume_1", pruned_1),
        ("pruned_resume_2", pruned_2),
        ("pruned_continue", pruned_continue),
        ("pruned_random_init", pruned_reset)
    ]

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

            with open(f"graphs/graph_data/reset-{run_id}-{ratio}.json", "w") as write_file:
                write_file.write(json.dumps(graph_data))
