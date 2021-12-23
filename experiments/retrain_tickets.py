import json
import os

import torch
from torch.nn import MSELoss
from torch.optim import Adam

from models.conv_ae import ConvAE
from procedures.test import test
from procedures.train import train
from settings.prune_settings import DRAW_PER_EPOCH, PRUNE_RATIOS
from settings.retrain_settings import RETRAIN_EPOCHS, RETRAIN_LR, RETRAIN_RESUME_EPOCH, RETRAIN_L2REG
from settings.train_settings import DRAW_EPOCHS, NETWORK

from utils.file import change_working_dir
from datasets.get_loaders import get_loaders
from utils.get_run_id import last_run

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
change_working_dir()
run_id = last_run()
checkpoint_folder = f"runs/{run_id}/"
graph_data_folder = f"graphs/graph_data/{run_id}"

train_loader, test_loader = get_loaders()
criterion = MSELoss()

skip_epochs = 4
train_every = 4

# overrides

if __name__ == '__main__':
    if os.path.exists(f"graphs/graph_data/{run_id}.json"):
        with open(f"graphs/graph_data/{run_id}.json", "r") as read_file:
            graph_data = json.loads(read_file.read())
    else:
        graph_data = {}

    print(f"Retraining tickets of run {run_id}")
    print(f"Estimated time to complete: {round((len(PRUNE_RATIOS) * DRAW_EPOCHS / skip_epochs * DRAW_PER_EPOCH / train_every + 1)*RETRAIN_EPOCHS*14/60, 1)} minutes")
    print()

    for ratio in [None] + PRUNE_RATIOS:
        masks = [x for x in os.listdir(checkpoint_folder) if x.startswith(f"keep-{ratio}-")]
        for draw_epoch in range(1, DRAW_EPOCHS + 1, skip_epochs):
            for sub_epoch in range(1, DRAW_PER_EPOCH + 1):
                if ratio is None and (draw_epoch != 1 or sub_epoch != train_every):
                    continue
                if sub_epoch % train_every == 0:
                    print(f"Retraining: ratio {ratio}, epoch {draw_epoch}, sub-epoch {sub_epoch}")
                    current_graph_data = []
                    graph_data[f"{ratio}-{draw_epoch}-{sub_epoch}"] = current_graph_data
                    resume = RETRAIN_RESUME_EPOCH if ratio is not None else None
                    network = NETWORK.init_from_checkpoint(run_id, ratio, draw_epoch, sub_epoch, param_epoch=resume).to(device)
                    optimiser = Adam(network.parameters(), lr=RETRAIN_LR, weight_decay=RETRAIN_L2REG)

                    for epoch in range(RETRAIN_EPOCHS):
                        train_loss = train(network, optimiser, criterion, train_loader, device)
                        test_loss = test(network, criterion, test_loader, device)

                        print(f"{epoch}/{RETRAIN_EPOCHS}: {round(train_loss, 8)} & {round(test_loss, 8)}")
                        current_graph_data.append((epoch + (resume if resume else 0), train_loss, test_loss))

                        with open(f"graphs/graph_data/{run_id}.json", "w") as write_file:
                            write_file.write(json.dumps(graph_data))

                        if test_loss - train_loss > 0.003 and train_loss < 0.017:
                            print("Aborting due to overfit")
                            break
