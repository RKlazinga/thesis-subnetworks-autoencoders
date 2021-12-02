import json
import os

import torch
from torch.nn import MSELoss
from torch.optim import Adam

from models.conv_ae import ConvAE
from procedures.test import test
from procedures.train import train
from settings.prune_settings import DRAW_PER_EPOCH, PRUNE_RATIOS

from settings.train_settings import *
from utils.ensure_correct_folder import change_working_dir
from utils.training_setup import get_loaders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
change_working_dir()
run_id = "d374677b9"
ratio = 0.75
train_every = 4
checkpoint_folder = f"runs/{run_id}/"
graph_data_folder = f"graphs/graph_data/{run_id}"

train_loader, test_loader = get_loaders()
criterion = MSELoss()


if __name__ == '__main__':
    if os.path.exists(f"graphs/graph_data/{run_id}.json"):
        with open(f"graphs/graph_data/{run_id}.json", "r") as read_file:
            graph_data = json.loads(read_file.read())
    else:
        graph_data = {}

    print(f"Retraining tickets of run {run_id}")
    print(f"Estimated time to complete: {round((len(PRUNE_RATIOS) * 6 * DRAW_PER_EPOCH / train_every + 1)*RETRAIN_EPOCHS*14/60, 1)} minutes")
    print()

    for ratio in [None] + PRUNE_RATIOS:
        masks = [x for x in os.listdir(checkpoint_folder) if x.startswith(f"keep-{ratio}-")]
        for draw_epoch in range(6):
            for sub_epoch in range(1, DRAW_PER_EPOCH + 1):
                if ratio is None and (draw_epoch != 0 or sub_epoch != train_every):
                    continue
                if sub_epoch % train_every == 0:
                    print(f"Retraining: ratio {ratio}, epoch {draw_epoch}, sub-epoch {sub_epoch}")
                    current_graph_data = []
                    graph_data[f"{ratio}-{draw_epoch}-{sub_epoch}"] = current_graph_data
                    network = ConvAE.init_from_checkpoint(run_id, ratio, draw_epoch, sub_epoch).to(device)
                    optimiser = Adam(network.parameters(), lr=LR)

                    for epoch in range(RETRAIN_EPOCHS):
                        train_loss = train(network, optimiser, criterion, train_loader, device)
                        test_loss = test(network, criterion, test_loader, device)

                        print(train_loss, test_loss)
                        current_graph_data.append((epoch, train_loss, test_loss))

                        with open(f"graphs/graph_data/{run_id}.json", "w") as write_file:
                            write_file.write(json.dumps(graph_data))
