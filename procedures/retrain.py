import json
import os.path

from torch.nn import MSELoss
from torch.optim import Adam

from datasets.get_loaders import get_loaders
from procedures.test import test
from procedures.train import train
from settings.retrain_settings import RETRAIN_LR, RETRAIN_EPOCHS, RETRAIN_L2REG
from utils.misc import get_device


def retrain_tagged_networks(networks, json_filename):
    device = get_device()
    graph_data = {}
    train_loader, test_loader = get_loaders()

    for net_tag, net in networks:
        print(f"Retraining: {net_tag}")
        current_graph_data = []
        graph_data[net_tag] = current_graph_data
        optimiser = Adam(net.parameters(), lr=RETRAIN_LR, weight_decay=RETRAIN_L2REG)
        criterion = MSELoss()

        for epoch in range(RETRAIN_EPOCHS):
            train_loss = train(net, optimiser, criterion, train_loader, device)
            test_loss = test(net, criterion, test_loader, device)

            print(train_loss, test_loss)
            current_graph_data.append((epoch, train_loss, test_loss))

            with open(json_filename, "w") as write_file:
                write_file.write(json.dumps(graph_data))


def retrain_with_shots(get_networks, json_filename, shots=3):
    if os.path.isfile(json_filename):
        with open(json_filename, "r") as readfile:
            graph_data = json.loads(readfile.read())
    else:
        graph_data = {}
    train_loader, test_loader = get_loaders()
    device = get_device()

    # multiple shots
    for shot in range(shots):
        nets = get_networks()
        for net_tag, net in nets:
            net.to(device)
            print(f"Shot {shot+1}/{shots}, retraining: {net_tag}")
            if net_tag not in graph_data:
                current_graph_data = {}
                graph_data[net_tag] = current_graph_data
            else:
                current_graph_data = graph_data[net_tag]
            optimiser = Adam(net.parameters(), lr=RETRAIN_LR, weight_decay=RETRAIN_L2REG)
            criterion = MSELoss()

            for epoch in range(RETRAIN_EPOCHS):
                train_loss = train(net, optimiser, criterion, train_loader, device)
                test_loss = test(net, criterion, test_loader, device)

                print(f"{epoch}/{RETRAIN_EPOCHS}: {train_loss}, {test_loss}")
                if epoch not in current_graph_data:
                    current_graph_data[epoch] = [[], []]
                current_graph_data[epoch][0].append(train_loss)
                current_graph_data[epoch][1].append(test_loss)

                with open(json_filename, "w") as write_file:
                    write_file.write(json.dumps(graph_data))
