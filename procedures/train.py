from tqdm import tqdm

from utils.channel_sparsity_reg import updateBN


def train(network, opt, criterion, train_loader):
    train_loss = 0
    network.train()
    for idx, batch in enumerate(tqdm(train_loader)):
        img, _ = batch
        target = img.clone()

        opt.zero_grad()
        reconstruction = network(img)

        single_loss = criterion(reconstruction, target)
        train_loss += single_loss.item()

        single_loss.backward()
        updateBN(network)

        opt.step()
    train_loss /= len(train_loader)
    return train_loss
