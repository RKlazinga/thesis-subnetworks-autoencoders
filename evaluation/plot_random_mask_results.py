import json
from typing import Dict
import matplotlib.pyplot as plt
from matplotlib import cm

from evaluation.plot_retrain_results import plot_single
from utils.file import change_working_dir
from utils.get_run_id import last_run
plt.rcParams["font.family"] = "serif"


def plot_acc_over_time_with_without_reinit(run_id, ratio):
    graph_data_file = f"graphs/graph_data/random_mask-{run_id}-{ratio}.json"

    cmap = cm.get_cmap("plasma")
    # colors = {
    #     "pruned": cmap(0.8),
    #     "pruned_reset": cmap(0.2),
    #     "unpruned": "grey",
    # }
    colors = {
        "unpruned": "grey",
        # "original_ticket": cmap(0),
        "original_ticket": cmap(0.25),
        # "reset_ticket": cmap(.4),
        "random_ticket": cmap(.7),
        # "random_ticket_resume": cmap(.85),
    }

    with open(graph_data_file, "r") as read_file:
        graph_data: Dict = json.loads(read_file.read())

        for key, data in graph_data.items():
            if key not in colors:
                print(f"WARNING: {key} not in defined colourmap")
                continue
            label = key.replace("_", " ").title()
            plot_single(data, colors[key], label, offset=int("Resume" in label))

        plt.title("Validating the importance of initialisation\n"
                  f"({round(100*(1-ratio))}% of channels pruned)")
        plt.ylabel("Test loss")
        plt.xlabel("Epoch")

        plt.gca().set_ylim([0.015, 0.025])
        plt.legend()
        plt.savefig(f"figures/retraining/random_mask-{run_id}-{ratio}.png", bbox_inches="tight")
        plt.show()


if __name__ == '__main__':
    change_working_dir()
    # _run_id = "[6, 4, 6]-bbbac9959"
    _run_id = last_run()

    plot_acc_over_time_with_without_reinit(_run_id, 0.5)
