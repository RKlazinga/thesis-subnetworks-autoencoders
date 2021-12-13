import json
from typing import Dict
import matplotlib.pyplot as plt
from matplotlib import cm

from evaluation.plot_retrain_results import plot_single
from utils.file import change_working_dir


def plot_acc_over_time_pruning(run_id, ratio):
    graph_data_file = f"graphs/graph_data/prune_compare-{run_id}-{ratio}.json"

    colors = {
        "unpruned": "grey",
        "no_redist": "red",
        "prop_redist": "yellow",
        "similarity_redist": "green",
    }

    with open(graph_data_file, "r") as read_file:
        graph_data: Dict = json.loads(read_file.read())

        for key, data in graph_data.items():
            label = key.replace("_", " ").title()
            plot_single(data, colors[key], label)

        plt.gca().set_ylim([0.015, 0.025])
        plt.legend()
        plt.savefig(f"graphs/reset-{run_id}-{ratio}.png", bbox_inches="tight")
        plt.show()


if __name__ == '__main__':
    change_working_dir()
    _run_id = "[6, 4, 6]-425222fd6"

    plot_acc_over_time_pruning(_run_id, 0.5)
