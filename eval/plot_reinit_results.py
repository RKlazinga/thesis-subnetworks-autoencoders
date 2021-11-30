import json
from typing import Dict
from matplotlib import cm
import matplotlib.pyplot as plt

from eval.plot_retrain_results import plot_single
from utils.ensure_correct_folder import change_working_dir


def plot_acc_over_time_multiple_drawings(run_id, ratio):
    graph_data_file = f"graphs/graph_data/reset-{run_id}-{ratio}.json"

    colors = {
        "unpruned": "grey",
        "pruned": "green",
        "pruned_reset": "red"
    }

    with open(graph_data_file, "r") as read_file:
        graph_data: Dict = json.loads(read_file.read())

        for key, data in graph_data.items():
            plot_single(data, colors[key])

        plt.gca().set_ylim([0.015, 0.035])
        plt.savefig(f"graphs/reset-{run_id}-{ratio}.png", bbox_inches="tight")
        plt.show()


if __name__ == '__main__':
    change_working_dir()
    _run_id = "d374677b9"

    plot_acc_over_time_multiple_drawings(_run_id, 0.75)
