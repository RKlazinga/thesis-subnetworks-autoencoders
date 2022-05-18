import torch
from tqdm import tqdm

from utils.misc import dev


def test(network, criterion, test_loader):
    test_loss = 0
    # we do not set the model to eval mode (as is typical)
    # this is because BN seems to behave better if you manually tell it to stop tracking running variables
    # this is done in progressive_mask_drawing.py
    # of course we do still use torch.no_grad for speed
    # network.eval()
    for idx, batch in enumerate(tqdm(test_loader)):
        if isinstance(batch, list):
            batch = batch[0]
        img = batch.to(dev())

        target = img.clone()
        with torch.no_grad():
            reconstruction = network(img)

        single_loss = criterion(reconstruction, target)
        test_loss += single_loss.item()
    test_loss /= len(test_loader)
    return test_loss
