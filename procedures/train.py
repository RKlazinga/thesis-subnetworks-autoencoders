from tqdm import tqdm

from settings.s import Settings
from utils.channel_sparsity_reg import update_bn


def train(network, opt, criterion, train_loader, device, prune_snapshot_method=None):
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
        img = batch.to(device)
        target = img.clone()

        opt.zero_grad()
        reconstruction = network(img)

        single_loss = criterion(reconstruction, target)
        train_loss += single_loss.item()

        single_loss.backward()

        # only add the regularisation if we are in the prune-snapshotting phase
        if prune_snapshot_method:
            update_bn(network)

        opt.step()
    train_loss /= train_count
    return train_loss
