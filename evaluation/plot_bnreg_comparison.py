import json
from typing import Dict
import matplotlib.pyplot as plt
from matplotlib import cm

from evaluation.plot_retrain_results import plot_single
from utils.file import change_working_dir
from utils.get_run_id import last_run
plt.rcParams["font.family"] = "serif"


def plot_acc_over_time_with_without_reinit(run_id, ratio):
    graph_data_file = f"graphs/graph_data/bnreg-0.7.json"

    cmap = cm.get_cmap("plasma")

    with open(graph_data_file, "r") as read_file:
        graph_data = json.loads(read_file.read())
        graph_data = sorted(list(graph_data.items()), key=lambda x: float(x[0]))

        for key, data in graph_data:
            if key == "pruned_continue": continue
            label = key.replace("_", " ").title()
            plot_single(data, None, label)
            print(f"Lowest of {key}: {min([x[2] for x in data])}")

        plt.gca().set_ylim([0.015, 0.025])
        plt.legend()
        plt.savefig(f"graphs/reset-{run_id}-{ratio}.png", bbox_inches="tight")
        plt.show()


if __name__ == '__main__':
    change_working_dir()
    _run_id = "[6, 4, 6]-425222fd6" #last_run()

    plot_acc_over_time_with_without_reinit(_run_id, 0.5)
