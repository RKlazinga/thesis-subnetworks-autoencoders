import matplotlib.pyplot as plt
from torch.nn import BatchNorm1d
from torch.nn.modules.batchnorm import _BatchNorm

from models.conv_ae import ConvAE
from settings.data_settings import NORMAL_STD_DEV, FLAT_DATAPOINTS
from settings.train_settings import NETWORK, SPARSITY_PENALTY
from utils.file import get_topology_of_run, get_epochs_of_run
from utils.get_run_id import last_run

plt.rcParams["font.family"] = "serif"


def analyse_at_epoch(run_id, epoch):
    model = NETWORK.init_from_checkpoint(run_id, None, None, None, param_epoch=epoch)
    latent_size, hidden_count, _ = get_topology_of_run(run_id)
    counter = 0
    for m in model.modules():
        if isinstance(m, _BatchNorm):
            counter += 1
            if counter == hidden_count + 1 + int(NETWORK == ConvAE):
                if isinstance(m, BatchNorm1d) and m.weight.data.shape[0] == latent_size:
                    return [x.item() for x in m.weight.data]
                else:
                    print("Did not find latent layer at expected position")


def plot_analysis_over_time(run_id):
    print(get_epochs_of_run(run_id))
    epochs = get_epochs_of_run(run_id)
    weights = [analyse_at_epoch(run_id, e) for e in range(1, epochs + 1)]
    weights = list(zip(*weights))

    for w in weights:
        plt.plot(range(1, epochs+1), w)

    plt.grid(True, linestyle="dashed")
    plt.yscale("log")

    plt.ylabel("Weight of latent neuron", labelpad=5)
    plt.gca().set_xlim([-1, epochs + 1])
    plt.gca().xaxis.get_major_locator().set_params(integer=True)
    plt.xlabel("Epoch")
    plt.title(f"{run_id} {NORMAL_STD_DEV} {SPARSITY_PENALTY}")

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    _run_id = last_run()

    # 0.05
    # _run_id = "newflat-[4, 2, 1]-34d2d723f"

    # 0.5
    # _run_id = "newflat-[4, 2, 1]-ab8e85518"

    print(_run_id)
    plot_analysis_over_time(_run_id)
    # analyse_at_epoch(_run_id, 2)
    # analyse_at_epoch(_run_id, 10)
    # analyse_at_epoch(_run_id, 20)
