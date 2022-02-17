import json
from typing import Dict
from matplotlib import cm
import matplotlib.pyplot as plt

from utils.file import change_working_dir
from utils.get_run_id import last_run
plt.rcParams["font.family"] = "serif"


def plot_single(data, color, label=None, offset=0, **kwargs):
    if isinstance(data, list):
        xs = [x[0]+offset for x in data]
        ys = [x[2] for x in data]
        plt.plot(xs, ys, color=color, label=label, **kwargs)
    elif isinstance(data, dict):
        xs = [int(x)+offset for x in data.keys()]
        ys = [x[1] for x in data.values()]
        y_avgs = [sum(y)/len(y) for y in ys]
        y_min = [min(y) for y in ys]
        y_max = [max(y) for y in ys]
        plt.plot(xs, y_avgs, color=color, label=label, **kwargs)
        plt.fill_between(xs, y_min, y_max, color=color, alpha=0.2)
    else:
        raise TypeError("Unknown data format")


def plot_acc_over_time_multiple_drawings(run_id, ratio):
    graph_data_file = f"graph_data/retraining/{run_id}.json"

    cmap = cm.get_cmap("plasma")
    with open(graph_data_file, "r") as read_file:
        graph_data: Dict = json.loads(read_file.read())

        # plot the various pruned runs of this ratio
        relevant_keys = [x for x in graph_data.keys() if x.startswith(f"{ratio}-")]
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
                  f"({round(100*ratio)}% of channels pruned)")
        plt.ylabel("Test loss")
        plt.xlabel("Epoch")
        plt.legend(loc="lower left")
        plt.savefig(f"figures/retraining/{run_id}-{ratio}.png", bbox_inches="tight")
        plt.show()


if __name__ == '__main__':
    change_working_dir()
    _run_id = "[6, 4, 6]-bbbac9959"
    # _run_id = last_run()

    # plot_acc_over_time_multiple_drawings(_run_id, 0.7)
    # plot_acc_over_time_multiple_drawings(_run_id, 0.7)
    plot_acc_over_time_multiple_drawings(_run_id, 0.5)
    # plot_acc_over_time_multiple_drawings(_run_id, 0.3)
