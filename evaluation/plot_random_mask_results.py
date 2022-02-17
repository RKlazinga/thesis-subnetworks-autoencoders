import json
from typing import Dict
import matplotlib.pyplot as plt
from matplotlib import cm

from evaluation.plot_retrain_results import plot_single
from utils.file import change_working_dir
from utils.get_run_id import last_run
plt.rcParams["font.family"] = "serif"


def plot(run_id, ratio):
    graph_data_file = f"graph_data/retraining/random_mask-{run_id}-{ratio}.json"

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
        "original_ticket_eb": cmap(0.25),
        # "reset_ticket": cmap(.4),
        "random_ticket": cmap(.7),
        # "random_ticket_resume": cmap(.85),
    }

    with open(graph_data_file, "r") as read_file:
        graph_data: Dict = json.loads(read_file.read())

        if any([key not in colors for key in graph_data.keys()]):
            print("Auto-generating colourmap")
            colors = {key: cmap(i/len(graph_data)) for i, key in enumerate(graph_data.keys())}

        for key, data in graph_data.items():
            if key not in colors:
                print(f"WARNING: {key} not in defined colourmap")
                continue

            if "eb" in key:
                continue
            label = key.replace("_", " ").title()
            plot_single(data, colors[key], label)

        plt.title("Validating the importance of initialisation\n"
                  f"({round(100*ratio)}% of channels pruned)")
        plt.ylabel("Test loss")
        plt.xlabel("Epoch")
        plt.grid(True, linestyle="dashed")

        plt.gca().set_ylim([0.015, 0.025])
        plt.legend()
        plt.savefig(f"figures/retraining/random_mask-{run_id}-{ratio}.png", bbox_inches="tight")
        plt.show()


if __name__ == '__main__':
    change_working_dir()
    # _run_id = "[6, 4, 6]-bbbac9959"
    _run_id = last_run()

    plot(_run_id, 0.5)