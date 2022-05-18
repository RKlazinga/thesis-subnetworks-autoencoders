import torch
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from torchvision.transforms import ToPILImage

from datasets.synthetic.im_generators import sine
from settings import Settings
from utils.file import change_working_dir

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 18


if __name__ == '__main__':
    d = [30, 100]
    # build images
    im_size = 28
    step = 7
    start = int(d[0] * 1e6)
    end = int(d[1] * 1e6)
    step = int((d[1] - d[0]) * 1e6 / step)
    ims = []
    xs = []
    for idx, v in enumerate(range(start, end, step)):
        variables = [v / 1e6]
        mean = torch.tensor([sine(x, y, *variables) for x in range(im_size) for y in range(im_size)]).view(
            (im_size, im_size))
        std = torch.full((im_size, im_size), Settings.NORMAL_STD_DEV)
        im_t = torch.clamp(torch.normal(mean, std).float(), 0, 1).view((1, im_size, im_size))
        ims.append(ToPILImage()(im_t))
        im = ToPILImage()(im_t)
        xs.append(v/1e6)
        plt.scatter(v/1e6, 0)
        plt.gca().add_artist(AnnotationBbox(OffsetImage(im, zoom=1.0), (v/1e6, 0), frameon=False))
    plt.gca().axes.get_yaxis().set_visible(False)
    plt.gca().set_xticks(xs)
    plt.gca().set_aspect(60)
    plt.xlabel("Horizontal Frequency")
    plt.tight_layout()
    change_working_dir()
    plt.savefig("../thsis-images/1a.png")
    im = Image.open("../thsis-images/1a.png")
    im = im.crop((25, int(im.height/2.8+30), im.width-20, int(im.height - im.height/2.8 + 30)))
    im.save("../thsis-images/1a.png")
    im.show()