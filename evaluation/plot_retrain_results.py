import json
from typing import Dict
from matplotlib import cm
import matplotlib.pyplot as plt

from utils.file import change_working_dir
from utils.get_run_id import last_run
plt.rcParams["font.family"] = "serif"


def plot_single(data, color, label=None, **kwargs):
    if isinstance(data, list):
        xs = [x[0] for x in data]
        ys = [x[2] for x in data]
    elif isinstance(data, dict):
        xs = list(data.keys())
        ys = [x[1] for x in data.values()]
    else:
        raise TypeError("Unknown data format")
    plt.plot(xs, ys, color=color, label=label, **kwargs)


def plot_acc_over_time_multiple_drawings(run_id, ratio):
    graph_data_file = f"graphs/graph_data/{run_id}.json"

    cmap = cm.get_cmap("plasma")
    with open(graph_data_file, "r") as read_file:
        graph_data: Dict = json.loads(read_file.read())

        # plot the various pruned runs of this ratio
        relevant_keys = [x for x in graph_data.keys() if x.startswith(f"{ratio}-")][::3]
        for idx, key in enumerate(relevant_keys):
            label = f"Pruned, drawn from epoch {key.split('-')[1]}"
            color = cmap(idx / len(relevant_keys))
            plot_single(graph_data[key], color, label)

        # CONV
        plt.gca().set_ylim([0.016, 0.028])
        # FF
        # plt.gca().set_ylim([0.20, 0.30])

        # always plot the unpruned retraining run in grey
        unpruned_key = [x for x in graph_data.keys() if x.startswith("None")][0]
        unpruned_data = graph_data[unpruned_key]
        plot_single(unpruned_data, "grey", "Unpruned", linewidth=2)

        plt.title("Training of lottery tickets vs unpruned network\n"
                  f"({round(100*(1-ratio))}% of channels pruned)")
        plt.ylabel("Test loss")
        plt.xlabel("Epoch")
        plt.legend(loc="lower left")
        plt.savefig(f"graphs/{run_id}-{ratio}.png", bbox_inches="tight")
        plt.show()


if __name__ == '__main__':
    change_working_dir()
    _run_id = "[6, 4, 6]-bbbac9959"
    # _run_id = last_run()

    # plot_acc_over_time_multiple_drawings(_run_id, 0.7)
    # plot_acc_over_time_multiple_drawings(_run_id, 0.7)
    plot_acc_over_time_multiple_drawings(_run_id, 0.5)
    # plot_acc_over_time_multiple_drawings(_run_id, 0.3)
