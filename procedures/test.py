import torch
from tqdm import tqdm


def test(network, criterion, test_loader, device):
    test_loss = 0
    network.eval()
    for idx, batch in enumerate(tqdm(test_loader)):
        img = batch[0].to(device)

        target = img.clone()
        with torch.no_grad():
            reconstruction = network(img)

        single_loss = criterion(reconstruction, target)
        test_loss += single_loss.item()
    test_loss /= len(test_loader)
    return test_loss