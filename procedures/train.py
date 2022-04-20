from tqdm import tqdm

from settings import Settings
from utils.channel_sparsity_reg import update_bn
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
        # scalar = (single_loss.item() - Settings.MIN_LOSS) / (Settings.MAX_LOSS - Settings.MIN_LOSS)
        # Settings.REG_MULTIPLIER = max(0.01, 1 - scalar)
        train_loss += single_loss.item()

        single_loss.backward()

        # only add the regularisation if we are in the prune-snapshotting phase
        if prune_snapshot_method:
            update_bn(network)

        opt.step()
    train_loss /= train_count
    return train_loss
