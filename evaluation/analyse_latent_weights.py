import json
from statistics import NormalDist
from typing import Union, List

import matplotlib.pyplot as plt
import torch
from torch.nn import BatchNorm1d
from torch.nn.modules.batchnorm import _BatchNorm

from settings import Settings
from utils.file import get_topology_of_run, get_epochs_of_run
from utils.get_run_id import last_run

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 18
LINE_W = 2


def find_latent_bn(model, topology=None, run_id=None):
    assert topology or run_id, "Either topology settings or run ID needs to be passed"
    if topology:
        latent_size, hidden_count = topology[:2]
    else:
        latent_size, hidden_count = get_topology_of_run(run_id)[:2]
    counter = 0
    for m in model.modules():
        if isinstance(m, _BatchNorm):
            counter += 1
            if isinstance(m, BatchNorm1d) and m.weight.data.shape[0] == latent_size:
                linear_layers = 2 if not hasattr(model, "linear_layers") else len(model.linear_layers)
                if counter in range(hidden_count + 1, hidden_count + linear_layers):
                    return m
    raise ValueError("Could not find latent neurons")


def analyse_all(run_id, epoch):
    model = Settings.NETWORK.init_from_checkpoint(run_id, None, None, None, param_epoch=epoch)
    weights = []
    for m in model.modules():
        if isinstance(m, _BatchNorm):
            weights.append((m, m.weight.data.tolist()))

    thresh = 2e-4
    for m, w in weights:
        zeros = len([0 for w2 in w if abs(w2) < thresh])
        print(f"{m.__class__.__name__} {len(w)-zeros}/{len(w)} remaining ({(1 - round(zeros / len(w), 3)) * 100}%)")


def analyse_at_epoch(run_id, epoch):
    model = Settings.NETWORK.init_from_checkpoint(run_id, None, None, None, param_epoch=epoch)
    m = find_latent_bn(model, run_id=run_id)
    return torch.abs(m.weight.data).tolist(), m.bias.data.tolist(), m.running_mean.tolist(), m.running_var.tolist()


def plot_analysis_over_time(run_id):
    epochs = get_epochs_of_run(run_id)
    latent_count = get_topology_of_run(run_id)[0]
    weights = [tuple([1 for _ in range(latent_count)])] + [analyse_at_epoch(run_id, e)[0] for e in range(1, epochs + 1)]
    weights = list(zip(*weights))

    upper_lim = 1e-1
    for w in weights:
        plt.plot(range(0, epochs+1), w, linewidth=LINE_W)
        upper_lim = max(upper_lim, max(w))

    plt.grid(True, linestyle="dashed")
    plt.yscale("log")

    plt.ylabel("Weight of latent neuron", labelpad=5)
    plt.gca().set_xlim([-1, epochs + 1])
    plt.gca().set_ylim([1e-5, 2 * upper_lim])
    plt.gca().xaxis.get_major_locator().set_params(integer=True)
    plt.xlabel("Epoch")

    plt.tight_layout()
    plt.savefig(f"figures/latent_strengths/{run_id}.png")

    plt.title(f"{run_id}\nstd:{Settings.NORMAL_STD_DEV} ({Settings.CONV_SPARSITY_PENALTY},{Settings.LINEAR_SPARSITY_PENALTY},{Settings.LATENT_SPARSITY_PENALTY})")

    plt.tight_layout()
    plt.show()


def plot_latent_count_over_time(run_ids: Union[str, List[str]], thresh=2e-4, show=True):
    if isinstance(run_ids, str):
        run_ids = [run_ids]

    if show:
        plt.grid(True, linestyle="dashed")
        ax1 = plt.gca()
        ax2 = ax1.twinx()

    for idx, run_id in enumerate(run_ids):
        epochs = get_epochs_of_run(run_id)
        latent_count = get_topology_of_run(run_id)[0]

        # get latent weights
        weights_biases = [analyse_at_epoch(run_id, e)[:2] for e in range(1, epochs + 1)]
        active_weight_count = [latent_count]
        for w, b in weights_biases:
            active_weight_count.append(0)
            for w2, b2 in zip(w, b):
                if abs(w2) > thresh:
                    dist = (NormalDist() * w2) + b2
                    if 1 - dist.cdf(0) > 0.01:
                        active_weight_count[-1] += 1

        # weights = [latent_count] + [len([w2 for w2 in w if abs(w2) > thresh]) + (idx - len(run_ids)//2)/20 for w in weights]

        # get loss curve
        with open(f"{Settings.RUN_FOLDER}/{run_id}/loss_graph.json", "r") as readfile:
            data = json.loads(readfile.read())
            xs = [d[0] for d in data]
            ys = [d[2] for d in data]

        # plot both
        if show:
            ax1.plot(range(0, epochs+1), active_weight_count, label="Active latent neurons", alpha=1.0, linewidth=2, snap=True)
            ax2.plot(xs, ys, color="red", label="Loss")
        # or return them
        else:
            return active_weight_count, ys

    ax1.yaxis.get_major_locator().set_params(integer=True)
    ax1.set_ylabel("Number of active latent neurons", labelpad=5)
    ax1.set_xlim([-1, epochs + 1])
    ax1.set_ylim(bottom=-0.25)
    ax2.set_ylim(bottom=0)
    ax2.set_ylabel("Loss")
    ax1.xaxis.get_major_locator().set_params(integer=True)

    ax1.set_xlabel("Epoch")
    # if len(run_ids) > 1:
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    plt.legend(h1+h2, l1+l2, loc='upper right')

    plt.tight_layout()
    plt.savefig(f"figures/latent_count/{run_id}.png")

    plt.title(f"{run_id}\nstd:{Settings.NORMAL_STD_DEV} ({Settings.CONV_SPARSITY_PENALTY},{Settings.LINEAR_SPARSITY_PENALTY},{Settings.LATENT_SPARSITY_PENALTY})")

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    _run_id = last_run()
    plot_analysis_over_time(_run_id)
    plot_latent_count_over_time(_run_id)
    # figure_of_runs(runs, plot_type="s", label="starter", captioner=lambda x: f"$\lambda={x[1:].split(']')[0]}$")


    # mult, bias = analyse_at_epoch(last_runs(1, 3)[0], 15)[:2]
    # for i, (m, b) in enumerate(zip(mult, bias)):
    #     if m == 0:
    #         print(f"{i}) ZERO")
    #     else:
    #         dist = (NormalDist() * m) + b
    #         print(f"{i}) weight {abs(m) > 2e-4} nonclipped {math.ceil(1 - dist.cdf(0) - 1e-3)}")
    # print(torch.log10(torch.abs(torch.Tensor(analyse_at_epoch(_run_id, 14)[0]))))
    # analyse_all(_run_id, 5)
    # analyse_all(_run_id, 26)
    # print()
    # analyse_all(_run_id, 20)

    # plot_latent_count_over_time(_run_id)
    # plot_latent_count_over_time(all_runs_matching("threevar2")[0])
    # analyse_all(all_runs_matching("e2d4")[0], 20)
    # plot_latent_count_over_time(last_runs(count=3, offset=0))