import torch
from matplotlib import pyplot as plt, cm
from torch.nn.modules.batchnorm import _BatchNorm

from models.conv_ae import ConvAE
from utils.file import change_working_dir, get_topology_of_run, get_params_of_run
from utils.get_run_id import last_run
from utils.misc import get_device

plt.rcParams["font.family"] = "serif"


def histogram_of_weights(run_id, epoch):
    device = get_device()
    net = ConvAE(*get_topology_of_run(run_id)).to(device)
    net.load_state_dict(get_params_of_run(run_id, epoch=epoch, device=device))

    lim = 0.8

    # get batchnorm layers
    bn = [x for x in net.modules() if isinstance(x, _BatchNorm)]

    # get absolute weight data of all batchnorm layers
    weights_per_bn = [torch.flatten(torch.abs(x.weight.data)) for x in bn]
    for i in range(len(weights_per_bn)):
        weights_per_bn[i] = [min(lim, x.item()) for x in weights_per_bn[i]]
    all_weights = []
    [all_weights.extend(x) for x in weights_per_bn]

    cmap = cm.get_cmap("inferno")
    colors = [cmap(i / len(weights_per_bn)) for i in range(len(weights_per_bn))]

    # the histogram of the data
    bins = [(x * 0.02) for x in range(40)]

    per_layer = False

    if per_layer:
        binheights, binwidths, _ = plt.hist(weights_per_bn, bins=bins, density=True, color=colors, alpha=0.75)

        # remove the plot and add lines instead
        plt.clf()

        for h, c in zip(binheights, colors):
            plt.plot(bins[:-1], h, color=c)
    else:
        plt.hist(all_weights, bins=bins, density=True, color="navy")

    plt.xlabel('Weight')
    plt.ylabel('Cumulative Density')
    plt.title('Histogram of BatchNorm weight parameters')
    plt.grid(True, linestyle="dashed")
    plt.savefig(f"figures/weight_histogram/{run_id}-E{epoch}.png")

    plt.show()


if __name__ == '__main__':

    change_working_dir()
    _run_id = last_run()

    # _run_id = "[6, 4, 6]-fe4bc8144"
    histogram_of_weights(_run_id, 12)
