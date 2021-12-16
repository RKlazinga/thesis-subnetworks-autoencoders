import json
from typing import Dict
from matplotlib import cm
import matplotlib.pyplot as plt

from utils.file import change_working_dir
from utils.get_run_id import last_run


def plot_single(data, color, label=None):
    xs = [x[0] for x in data]
    ys = [x[2] for x in data]
    plt.plot(xs, ys, color=color, label=label)


def plot_acc_over_time_multiple_drawings(run_id, ratio):
    graph_data_file = f"graphs/graph_data/{run_id}.json"

    cmap = cm.get_cmap("plasma")
    with open(graph_data_file, "r") as read_file:
        graph_data: Dict = json.loads(read_file.read())

        # always plot the unpruned retraining run in grey
        unpruned_key = [x for x in graph_data.keys() if x.startswith("None")][0]
        unpruned_data = graph_data[unpruned_key]
        plot_single(unpruned_data, "grey", "Unpruned")

        # plot the various pruned runs of this ratio
        relevant_keys = [x for x in graph_data.keys() if x.startswith(f"{ratio}-")]
        for idx, key in enumerate(relevant_keys):
            label = f"Drawn from epoch {key.split('-')[1]}"
            color = cmap(idx / len(relevant_keys))
            plot_single(graph_data[key], color, label)

        plt.gca().set_ylim([0.015, 0.035])
        plt.legend()
        plt.savefig(f"graphs/{run_id}-{ratio}.png", bbox_inches="tight")
        plt.show()


if __name__ == '__main__':
    change_working_dir()
    _run_id = last_run()

    plot_acc_over_time_multiple_drawings(_run_id, 0.9)
    plot_acc_over_time_multiple_drawings(_run_id, 0.7)
    plot_acc_over_time_multiple_drawings(_run_id, 0.5)
    plot_acc_over_time_multiple_drawings(_run_id, 0.3)
