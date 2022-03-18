import json
from typing import Union, List

import matplotlib.pyplot as plt
from torch.nn import BatchNorm1d
from torch.nn.modules.batchnorm import _BatchNorm
from tqdm import tqdm

from settings.data_settings import NORMAL_STD_DEV
from settings.global_settings import RUN_FOLDER
from settings.train_settings import NETWORK, SPARSITY_PENALTY, LATENT_SPARSITY_PENALTY
from utils.file import get_topology_of_run, get_epochs_of_run
from utils.get_run_id import last_run, last_runs, all_runs_matching

plt.rcParams["font.family"] = "serif"


def analyse_at_epoch(run_id, epoch):
    model = NETWORK.init_from_checkpoint(run_id, None, None, None, param_epoch=epoch)
    latent_size, hidden_count, _ = get_topology_of_run(run_id)
    counter = 0
    for m in model.modules():
        if isinstance(m, _BatchNorm):
            counter += 1
            if isinstance(m, BatchNorm1d) and m.weight.data.shape[0] == latent_size:
                if counter in range(hidden_count + 1, hidden_count + len(model.linear_layers)):
                    return m.weight.data.tolist(), m.running_mean.tolist(), m.running_var.tolist()
    raise ValueError("Could not find latent neurons")


def plot_analysis_over_time(run_id):
    epochs = get_epochs_of_run(run_id)
    latent_count = get_topology_of_run(run_id)[0]
    weights = [tuple([0.1 for _ in range(latent_count)])] + [analyse_at_epoch(run_id, e)[0] for e in range(1, epochs + 1)]
    weights = list(zip(*weights))

    for w in weights:
        plt.plot(range(0, epochs+1), w)

    plt.grid(True, linestyle="dashed")
    plt.yscale("log")

    plt.ylabel("Weight of latent neuron", labelpad=5)
    plt.gca().set_xlim([-1, epochs + 1])
    plt.gca().xaxis.get_major_locator().set_params(integer=True)
    plt.xlabel("Epoch")
    plt.title(f"{run_id} std:{NORMAL_STD_DEV} reg:{SPARSITY_PENALTY} latent_reg:{LATENT_SPARSITY_PENALTY}")

    plt.tight_layout()
    plt.show()


def plot_latent_count_over_time(run_ids: Union[str, List[str]], thresh=2e-4, show=True):
    if isinstance(run_ids, str):
        run_ids = [run_ids]

    plt.grid(True, linestyle="dashed")
    plt.ylabel("Number of active latent neurons", labelpad=5)
    ax1 = plt.gca()
    ax2 = ax1.twinx()

    for idx, run_id in enumerate(run_ids):
        epochs = get_epochs_of_run(run_id)
        latent_count = get_topology_of_run(run_id)[0]
        weights = [analyse_at_epoch(run_id, e)[0] for e in range(1, epochs + 1)]
        weights = [latent_count] + [len([w2 for w2 in w if w2 > thresh]) + (idx - len(run_ids)//2)/20 for w in weights]
        if show:
            ax1.plot(range(0, epochs+1), weights, label="Active latent neurons", alpha=1.0, linewidth=2, snap=True)

            with open(f"{RUN_FOLDER}/{run_id}/loss_graph.json", "r") as readfile:
                data = json.loads(readfile.read())
                xs = [d[0] for d in data]
                ys = [d[2] for d in data]
                ax2.plot(xs, ys, color="red", label="Loss")

    if not show:
        return weights

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
    # plt.legend()
    # plt.title(f"{run_id} std:{NORMAL_STD_DEV} reg:{SPARSITY_PENALTY} latent_reg:{LATENT_SPARSITY_PENALTY}")

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    _run_id = last_run()

    # plot_analysis_over_time(_run_id)
    plot_latent_count_over_time(all_runs_matching("[0.0001,0.0005,0.006]-newer")[0])
    plot_latent_count_over_time(all_runs_matching("[0.0001,0.005,0.006]-newer")[0])
    # plot_latent_count_over_time("[0.0001,0.05,0.006]-newer_synthim-[12, 5, 6]-9082a3e2c")
    # plot_latent_count_over_time("[0.0001,0.005,0.006]-newer_synthim-[12, 5, 6]-9db5763c8")
    # plot_latent_count_over_time("[0.0001,0.01,0.006]-newer_synthim-[12, 5, 6]-d3bccf2b8")
    # plot_latent_count_over_time(last_runs(count=3, offset=0))