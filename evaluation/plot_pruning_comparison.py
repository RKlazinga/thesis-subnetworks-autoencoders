import json
from typing import Dict
import matplotlib.pyplot as plt

from evaluation.plot_retrain_results import plot_single
from utils.file import change_working_dir
from utils.get_run_id import last_run
plt.rcParams["font.family"] = "serif"


def plot(run_id, ratio):
    graph_data_file = f"graph_data/retraining/prune_compare-{run_id}-{ratio}.json"

    colors = {
        "unpruned": "grey",
        "no_redist": "red",
        "no_redist_lim": "orange",
        "similarity_redist": "green",
        "similarity_redist1": "purple",
    }

    with open(graph_data_file, "r") as read_file:
        graph_data: Dict = json.loads(read_file.read())

        for key, data in graph_data.items():
            label = key.replace("_", " ").title()
            plot_single(data, colors[key], label)

        plt.gca().set_ylim([0.016, 0.022])
        plt.legend()
        plt.savefig(f"figures/retraining/prune_compare-{run_id}-{ratio}.png", bbox_inches="tight")
        plt.show()


if __name__ == '__main__':
    change_working_dir()
    _run_id = last_run()

    plot(_run_id, 0.5)
