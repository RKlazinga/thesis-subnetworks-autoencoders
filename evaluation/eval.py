import torch
from torchvision.transforms import ToPILImage
from PIL import Image


def eval_network(net, im, device):
    net.eval()
    with torch.no_grad():
        if isinstance(im, list):
            im = im[0]
        out = torch.clamp(net(im.to(device)), 0, 1)
    combined = Image.new("L", (56, 56))
    combined.paste(ToPILImage()(im[0]), (0, 0))
    combined.paste(ToPILImage()(out[0]), (28, 0))
    combined.show()
