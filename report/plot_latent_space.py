import json
import os
from typing import Union, Tuple, Iterable

import torch
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from torch.nn.modules.batchnorm import _BatchNorm
from torchvision.transforms import ToPILImage

from datasets.synthetic.im_generators import sine
from evaluation.analyse_latent_weights import analyse_at_epoch
from settings import Settings
from utils.file import change_working_dir

plt.rcParams["font.family"] = "serif"


def latent_scatterplot(run_id, generator, epoch, domains, step=12, imgs_on_points=False):
    model = Settings.NETWORK.init_from_checkpoint(run_id, None, None, param_epoch=epoch)
    model.eval()

    # print achieved loss for this encoding
    with open(os.path.join(Settings.RUN_FOLDER, run_id, "loss_graph.json"), "r") as readfile:
        g_data = json.loads(readfile.read())
    print(f"Loss attained: {g_data[-1][-1]}")

    # disable running stats since we will be inputting weird, small batches
    for m in model.modules():
        if isinstance(m, _BatchNorm):
            m.track_running_stats = False
    latent_weight, latent_bias, latent_mean, latent_var = analyse_at_epoch(run_id, epoch)

    # print(latent_bias)
    thresh = 10 ** (-2.5)
    neurons = [(i, w) for i, w in enumerate(latent_weight) if abs(w) > thresh]

    # build images
    im_size = 28
    for d in domains:
        start = int(d[0] * 1e6)
        end = int(d[1] * 1e6)
        step = int((d[1] - d[0]) * 1e6 / step)
        xs = []
        ims = []
        ys = [[] for _ in neurons]
        for idx, v in enumerate(range(start, end, step)):
            variables = [v / 1e6]
            mean = torch.tensor([generator(x, y, *variables) for x in range(im_size) for y in range(im_size)]).view(
                (im_size, im_size))
            std = torch.full((im_size, im_size), Settings.NORMAL_STD_DEV)
            im_t = torch.clamp(torch.normal(mean, std).float(), 0, 1).view((1, im_size, im_size))
            ims.append(ToPILImage()(im_t))
            with torch.no_grad():
                latent = model.encoder(im_t.view(1, 1, im_size, im_size)).tolist()[0]
            xs.append(v / 1e6)
            for i, (n, _) in enumerate(neurons):
                ys[i].append(latent[n])
        for y in ys:
            plt.scatter(xs, y)
            bigims = []
            if imgs_on_points:
                prev_y = -1e6
                next_y = iter(y[1:] + [0])

                for ix, iy, ny, im in zip(xs, y, next_y, ims):
                    if max(abs(iy - prev_y), abs(iy - ny)) > 0.025:
                        bigims.append((ix, iy, ny, im))
                    else:
                        plt.gca().add_artist(AnnotationBbox(OffsetImage(im, zoom=1.0), (ix, iy), frameon=False))
                    prev_y = iy
                for ix, iy, ny, im in bigims:
                    plt.gca().add_artist(AnnotationBbox(OffsetImage(im, zoom=1.25), (ix, iy), frameon=True))
        plt.xlabel("Horizontal Frequency of Image")
        plt.ylabel("Latent Value")
        plt.grid(True, linestyle="dashed")
        plt.tight_layout()
        change_working_dir()
        fname = f"latent_scatter_{len(neurons)}"
        if imgs_on_points:
            fname += "_withims"
        plt.savefig(f"figures/{fname}.png")


def plot_latent_space_2d(run_id, epoch, domain: Union[Tuple, Iterable[Tuple]] = (-3, 3), steps=12):
    if isinstance(domain, tuple):
        domain = [domain, domain]
    model = Settings.NETWORK.init_from_checkpoint(run_id, None, None, param_epoch=epoch)
    model.eval()

    # print achieved loss for this encoding
    with open(os.path.join(Settings.RUN_FOLDER, run_id, "loss_graph.json"), "r") as readfile:
        g_data = json.loads(readfile.read())
    print(f"Loss attained: {g_data[-1][-1]}")

    # disable running stats since we will be inputting weird, small batches
    for m in model.modules():
        if isinstance(m, _BatchNorm):
            m.track_running_stats = False
    latent_weight, latent_bias, latent_mean, latent_var = analyse_at_epoch(run_id, epoch)

    # print(latent_bias)
    thresh = 10**(-2.5)
    neurons = [(i, w) for i, w in enumerate(latent_weight) if abs(w) > thresh]

    if not 0 < len(neurons) < 3:
        raise ValueError(f"Relevant neuron count out of range: {neurons}")

    start_x = int(domain[0][0] * 1e6)
    end_x = int(domain[0][1] * 1e6)
    step_x = int((domain[0][1] - domain[0][0]) * 1e6 / steps)
    start_y = int(domain[1][0] * 1e6)
    end_y = int(domain[1][1] * 1e6)
    step_y = int((domain[1][1] - domain[1][0]) * 1e6 / steps)

    imsize = 28
    bg = Image.new("L", (imsize * steps, imsize * (1 if len(neurons) == 1 else steps)), color=0)
    for idx_x, x in enumerate(range(start_x, end_x, step_x)):
        for idx_y, y in enumerate(range(start_y, end_y, step_y)):
            if len(neurons) == 1 and idx_y > 0:
                continue
            # inp = [0 for _ in range(topology[0])]
            inp = latent_bias[:]
            for (idx, weight), v in zip(neurons, [x/1e6, y/1e6]):
                inp[idx] += v*weight
            inp = torch.tensor(inp).float().view(1, -1)
            out = torch.clamp(model.decoder(inp).detach().cpu(), 0, 1)
            out = ToPILImage()(out[0])
            bg.paste(out, (idx_x * imsize, idx_y * imsize))
    bg.show()


if __name__ == '__main__':
    change_working_dir()
    _epoch = 20

    # 1D good weather
    _run_id = "[8-1-0.001-0.001]-basic-synthim_1var-[8, 5, 12, 1]-e136e51ef"
    # plot_latent_space_2d(_run_id, _epoch, domain=(-0.5, 10), steps=12)
    # latent_scatterplot(_run_id, sine, _epoch, [(30, 100)], 30, imgs_on_points=False)
    # latent_scatterplot(_run_id, sine, _epoch, [(30, 100)], 30, imgs_on_points=True)

    # 1D bad weather (2 dims)
    _run_id = "[8-1-0.001-0.01]-basic-synthim_1var-[8, 5, 12, 1]-eafdda5d6"
    # plot_latent_space_2d(_run_id, _epoch, domain=(-4, 8), steps=12)
    # latent_scatterplot(_run_id, sine, _epoch, [[30, 100]], 30)




