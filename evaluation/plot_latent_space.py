import torch
from PIL import Image
from torch.nn.modules.batchnorm import _BatchNorm
from torchvision.transforms import ToPILImage

from evaluation.analyse_latent_weights import analyse_at_epoch
from settings.s import Settings
from utils.file import change_working_dir, get_topology_of_run
from utils.get_run_id import last_run


def plot_latent_space_2d(run_id, epoch, domain=(0, 1), steps=16):
    model = Settings.NETWORK.init_from_checkpoint(run_id, None, None, param_epoch=epoch)
    topology = get_topology_of_run(run_id)
    model.eval()

    # disable running stats since we will be inputting weird, small batches
    for m in model.modules():
        if isinstance(m, _BatchNorm):
            m.track_running_stats = False
    latent_weight, latent_bias, latent_mean, latent_var = analyse_at_epoch(run_id, epoch)

    thresh = 10**(-2.5)
    neurons = [i for i in range(len(latent_weight)) if latent_weight[i] > thresh]

    if not 0 < len(neurons) < 3:
        raise ValueError(f"Relevant neuron count out of range: {neurons}")

    start = int(domain[0] * 1e6)
    end = int(domain[1] * 1e6)
    step = int((domain[1] - domain[0]) * 1e6 / steps)

    imsize = 28
    bg = Image.new("L", (imsize * steps, imsize * steps), color=0)
    for idx_x, x in enumerate(range(start, end, step)):
        for idx_y, y in enumerate(range(start, end, step)):
            inp = [0 for _ in range(topology[0])]
            for idx, v in zip(neurons, [x/1e6, y/1e6]):
                inp[idx] = v
            inp = torch.tensor(inp).float().view(1, -1)
            out = model.decoder(inp)
            out = ToPILImage()(out[0])
            bg.paste(out, (idx_y * imsize, idx_x * imsize))
    bg.show()


if __name__ == '__main__':
    change_working_dir()
    # _run_id = last_run()
    _run_id = "[0.0001,0.005,0.006]-newer_synthim-[12, 5, 6]-9db5763c8"
    _epoch = 30
    plot_latent_space_2d(_run_id, _epoch)




