import json
import os

from torch.nn import MSELoss
from torch.optim import Adam

from procedures.test import test
from procedures.train import train
from settings import Settings
from utils.file import change_working_dir
from datasets.get_loaders import get_loaders
from utils.get_run_id import last_run
from utils.misc import dev


change_working_dir()
run_id = last_run()
checkpoint_folder = f"{Settings.RUN_FOLDER}/{run_id}/masks"

train_loader, test_loader = get_loaders()
criterion = MSELoss()

skip_epochs = 4
train_every = 4

if __name__ == '__main__':
    if os.path.exists(f"graph_data/retraining/{run_id}.json"):
        with open(f"graph_data/retraining/{run_id}.json", "r") as read_file:
            graph_data = json.loads(read_file.read())
    else:
        graph_data = {}

    print(f"Retraining tickets of run {run_id}")
    eta = round((len(Settings.PRUNE_RATIOS) * Settings.DRAW_EPOCHS / skip_epochs * Settings.DRAW_PER_EPOCH / train_every + 1)
                * Settings.RETRAIN_EPOCHS * 14 / 60, 1)
    print(f"Estimated time to complete: {eta} minutes")
    print()

    for ratio in [None] + Settings.PRUNE_RATIOS:
        masks = [x for x in os.listdir(checkpoint_folder) if x.startswith(f"prune-{ratio}-")]
        for draw_epoch in range(1, Settings.DRAW_EPOCHS + 1, skip_epochs):
            for sub_epoch in range(1, Settings.DRAW_PER_EPOCH + 1):
                if ratio is None and (draw_epoch != 1 or sub_epoch != train_every):
                    continue
                if sub_epoch % train_every == 0:
                    print(f"Retraining: ratio {ratio}, epoch {draw_epoch}, sub-epoch {sub_epoch}")
                    current_graph_data = []
                    graph_data[f"{ratio}-{draw_epoch}-{sub_epoch}"] = current_graph_data
                    resume = Settings.RETRAIN_RESUME_EPOCH if ratio is not None else None
                    network = Settings.NETWORK.init_from_checkpoint(run_id, ratio, draw_epoch, sub_epoch,
                                                           param_epoch=resume).to(dev())
                    optimiser = Adam(network.parameters(), lr=Settings.RETRAIN_LR, weight_decay=Settings.RETRAIN_L2REG)

                    for epoch in range(Settings.RETRAIN_EPOCHS):
                        train_loss = train(network, optimiser, criterion, train_loader)
                        test_loss = test(network, criterion, test_loader)

                        print(f"{epoch}/{Settings.RETRAIN_EPOCHS}: {round(train_loss, 8)} & {round(test_loss, 8)}")
                        current_graph_data.append((epoch + (resume if resume else 0), train_loss, test_loss))

                        with open(f"graph_data/retraining/{run_id}.json", "w") as write_file:
                            write_file.write(json.dumps(graph_data))
