from tqdm import tqdm

from datasets.dataset_options import DatasetOption
from models.ff_ae import FeedforwardAE
from settings import Settings
from utils.subgrad_l1 import update_bn
from utils.misc import dev


def train(network, opt, criterion, train_loader, prune_snapshot_method=None):
    train_loss = 0
    network.train()

    snapshot_counter = 1
    train_count = len(train_loader)
    for idx, batch in enumerate(tqdm(train_loader)):
        if prune_snapshot_method:
            if snapshot_counter / Settings.DRAW_PER_EPOCH <= (idx+1) / train_count:
                prune_snapshot_method(snapshot_counter)
                snapshot_counter += 1
        if isinstance(batch, list):
            batch = batch[0]
        img = batch.to(dev())
        target = img.clone()

        opt.zero_grad()
        reconstruction = network(img)

        single_loss = criterion(reconstruction, target)

        # multiply regularisation based on currently achieved loss
        if Settings.REG_SCALING_TYPE == "switch":
            if single_loss.item() < Settings.MAX_LOSS:
                Settings.REG_MULTIPLIER = 1
            else:
                Settings.REG_MULTIPLIER = 0
        elif Settings.REG_SCALING_TYPE == "linear":
            scalar = (single_loss.item() - Settings.MIN_LOSS) / (Settings.MAX_LOSS - Settings.MIN_LOSS)
            Settings.REG_MULTIPLIER = min(max(0, 1 - scalar), 1)
        elif Settings.REG_SCALING_TYPE == "multi":
            if single_loss.item() > Settings.MAX_LOSS:
                Settings.REG_MULTIPLIER = 0
            else:
                scalar = 1 / (single_loss.item() * 10)
                Settings.REG_MULTIPLIER = scalar
        elif Settings.REG_SCALING_TYPE in ["off", None]:
            pass
        else:
            raise KeyError(f"Unknown regularisation scaling type: {Settings.REG_SCALING_TYPE}")

        if Settings.REG_SCALING_TYPE not in ["off", None] and Settings.NETWORK == FeedforwardAE:
            raise ValueError("Regularisation scaling is not intended to be used in the 1D case")

        train_loss += single_loss.item()

        single_loss.backward()

        # only add the regularisation if we are in the prune-snapshotting phase
        if prune_snapshot_method:
            update_bn(network)

        opt.step()
    train_loss /= train_count
    return train_loss
