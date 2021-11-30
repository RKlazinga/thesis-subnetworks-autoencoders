import json
from typing import Dict
from matplotlib import cm
import matplotlib.pyplot as plt

from utils.ensure_correct_folder import change_working_dir


def plot_single(data, color):
    xs = [x[0] for x in data]
    ys = [x[2] for x in data]
    plt.plot(xs, ys, color=color)


def plot_acc_over_time_multiple_drawings(run_id, ratio):
    graph_data_file = f"graphs/graph_data/{run_id}.json"

    cmap = cm.get_cmap("plasma")
    with open(graph_data_file, "r") as read_file:
        graph_data: Dict = json.loads(read_file.read())

        # always plot the unpruned retraining run in grey
        unpruned_key = [x for x in graph_data.keys() if x.startswith("None")][0]
        unpruned_data = graph_data[unpruned_key]
        plot_single(unpruned_data, "grey")

        # plot the various pruned runs of this ratio
        relevant_keys = [x for x in graph_data.keys() if x.startswith(f"{ratio}-")]
        for idx, key in enumerate(relevant_keys):
            color = cmap(idx / len(relevant_keys))
            print(key, color)
            plot_single(graph_data[key], color)

        plt.gca().set_ylim([0.015, 0.035])
        plt.savefig(f"graphs/{run_id}-{ratio}.png", bbox_inches="tight")
        plt.show()



if __name__ == '__main__':
    change_working_dir()
    _run_id = "d374677b9"

    plot_acc_over_time_multiple_drawings(_run_id, 0.75)
