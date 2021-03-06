import json
from typing import Dict
import matplotlib.pyplot as plt
from matplotlib import cm

from evaluation.plot_retrain_results import plot_single
from utils.file import change_working_dir
from utils.get_run_id import last_run
plt.rcParams["font.family"] = "serif"


def plot(run_id, ratio):
    graph_data_file = f"graph_data/retraining/reset-{run_id}-{ratio}.json"

    cmap = cm.get_cmap("plasma")
    colors = {
        "unpruned": "grey",
        "pruned_no_resume": cmap(0),
        "pruned_resume_1": cmap(.4),
        "pruned_resume_2": cmap(.8),
        "pruned_continue": cmap(.99),
        "pruned_random_init": "red"
    }

    with open(graph_data_file, "r") as read_file:
        graph_data: Dict = json.loads(read_file.read())

        for key, data in graph_data.items():
            if key == "pruned_continue":
                continue
            label = key.replace("_", " ").title()
            plot_single(data, colors[key], label)

        plt.title("Validating the importance of initialisation\n"
                  f"({round(100*ratio)}% of channels pruned)")
        plt.ylabel("Test loss")
        plt.xlabel("Epoch")

        plt.gca().set_ylim([0.015, 0.025])
        plt.legend()
        plt.savefig(f"figures/retraining/reset-{run_id}-{ratio}.png", bbox_inches="tight")
        plt.show()


if __name__ == '__main__':
    change_working_dir()
    _run_id = last_run()

    plot(_run_id, 0.5)
