# tensor shape: batch x channels x width x height
# we want to compare _channels_ pairwise, across multiple batches
# for now, batch=1
import torch
from PIL import Image, ImageFont, ImageDraw
from torch.functional import F
from torch.nn import Conv2d
from torchvision.transforms import ToPILImage

from models.conv_ae import ConvAE
from utils.ensure_correct_folder import change_working_dir
from utils.training_setup import get_loaders

from settings.global_settings import RUN_FOLDER

BATCH, CHANNELS, WIDTH, HEIGHT = 2, 2, 3, 3


combined = torch.Tensor([
    [[
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
    ],
    [
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0],
    ]], [[
        [0, 1, 0],
        [0, 1, 0],
        [0, 1, 0],
    ], [
        [1, 0, 1],
        [0, 1, 0],
        [1, 0, 1],
    ]],
]).view(BATCH, CHANNELS, WIDTH, HEIGHT)


def pairwise_sim(tensor):
    sim = torch.zeros((tensor.shape[1], tensor.shape[1]))
    for i in range(tensor.shape[1]):
        x = tensor[:, i, :, :]
        for j in range(tensor.shape[1]):
            if i > j:
                y = tensor[:, j, :, :]
                similarity = final_sim(x, y)
                sim[i][j] = similarity
                sim[j][i] = similarity
    for i in range(tensor.shape[1]):
        sim[i, i] = 0
    # sim = torch.max(sim, dim=0).values - sim
    # # sim = torch.softmax(sim, dim=0)
    # # for i in range(tensor.shape[1]):
    # #     sim[i, i] = 0
    sim = torch.square(sim)
    print(sim.shape)
    sim = torch.div(sim, torch.sum(sim, dim=0))
    print(torch.sum(sim))
    return sim


def final_sim(a, b):
    return torch.min(torch.sum(torch.abs(a - b)),
                     torch.sum(torch.abs((1 - a) - b)))


def analyse_sim(sim, tensor):
    bg = Image.new("RGBA", (1800, 1000))
    draw = ImageDraw.ImageDraw(bg)

    def pil(t):
        t_pos = t.clone()
        t_pos[t_pos < 0] = 0
        return ToPILImage()(t_pos).resize((70, 70), Image.NEAREST)

    channel_to_vis = 9
    # for j in range(tensor.shape[0]):
    #     bg.paste(pil(tensor[j, channel_to_vis, :, :]), (j * 75, 0))

    for i in range(tensor.shape[1]):
        if i != channel_to_vis:
            draw.text((i*75, 90), f"diff={round(sim[channel_to_vis][i].item(), 4)}", fill=(0, 0, 0))
        else:
            draw.text((i*75, 90), "Compare", fill=(0, 0, 0))
        for j in range(tensor.shape[0]):
            bg.paste(pil(tensor[j, i, :, :]), (i * 75, j * 75 + 100))

    bg.show()


if __name__ == '__main__':
    change_working_dir()
    net = ConvAE(6, 4, 6)
    net.load_state_dict(torch.load(f"{RUN_FOLDER}/ebf369d1b/trained-3.pth"))
    net.eval()

    layer_to_process = 3

    def hook(module, _, output):
        tensor = output.clone()
        tensor /= torch.amax(tensor, dim=(2, 3), keepdim=True)
        global layer_to_process
        layer_to_process -= 1
        if layer_to_process <= 0:
            # the module has produced a BxCxWxH tensor
            # we can analyse the channels in this tensor pairwise
            sim = pairwise_sim(tensor)
            analyse_sim(sim, tensor)

            # remove all other hooks (for now)
            [h.remove() for h in handles]

    handles = []
    for m in net.modules():
        if isinstance(m, Conv2d):
            handles.append(m.register_forward_hook(hook))

    _, test_loader = get_loaders(batch_size=12)
    img, _ = next(iter(test_loader))
    with torch.no_grad():
        net(img)
