import json
import os

from matplotlib import pyplot as plt, cm, ticker

from utils.file import change_working_dir
from utils.get_run_id import last_run
plt.rcParams["font.family"] = "serif"


def plot(run_id, baseline=None):
    """
    Find all retraining runs with this run ID (multiple ratios).
    Take the smallest loss for each ratio and plot loss over ratio.
    """
    def as_percentage(v):
        if baseline is None:
            return v
        return 100 * (v / baseline - 1)

    # get all run files for this id
    run_files = [x for x in os.listdir("graph_data/retraining/") if x.startswith(run_id)]
    graph_data = {}
    for f in run_files:
        with open(f"graph_data/retraining/{f}") as read_file:
            ratio = float(f.removesuffix(".json").split("-")[-1])
            plot_json = json.loads(read_file.read())
            for line_name in plot_json.keys():
                if line_name not in graph_data:
                    graph_data[line_name] = [[], [], [], []]

                # simply take the last datapoint and average it (assume monotonous loss decrease)
                # second index is test loss (first is train)
                smallest_avg = 1e9
                smallest_min = 1e9
                smallest_max = 1e9
                for values in plot_json[line_name].values():
                    smallest_avg = min(smallest_avg, sum(values[1])/len(values[1]))
                    smallest_min = min(smallest_min, min(values[1]))
                    smallest_max = min(smallest_max, max(values[1]))

                graph_data[line_name][0].append(ratio)
                graph_data[line_name][1].append(as_percentage(smallest_avg))
                graph_data[line_name][2].append(as_percentage(smallest_min))
                graph_data[line_name][3].append(as_percentage(smallest_max))

    cmap = cm.get_cmap("plasma")
    colors = {
        "unpruned": "grey",
        "original_ticket": cmap(0.25),
        "random_ticket": cmap(.75),
        "utterly_random": cmap(.5),
    }

    xmin = 1e9
    xmax = -1e9
    for line_name, line_data in graph_data.items():
        xmin = min(xmin, *line_data[0])
        xmax = max(xmax, *line_data[0])
        plt.plot(line_data[0], line_data[1], color=colors[line_name], label=line_name.replace("_", " ").title())
        plt.fill_between(line_data[0], line_data[2], line_data[3], color=colors[line_name], alpha=0.2)

    plt.legend()
    plt.title("Validating the relevance of specific selected channels\n"
              f"compared to a random mask with equivalent pruning per layer")
    if baseline:
        plt.ylabel("Change in minimum test loss (%)")
        plt.gca().yaxis.set_major_formatter(ticker.PercentFormatter())
    else:
        plt.ylabel("Minimum Test Loss")
    plt.xlabel("Fraction of removed channels")
    plt.grid(True, linestyle="dashed")
    # plt.gca().set_xlim([0, 1])

    plt.savefig(f"figures/validation/loss_v_ratio-{run_id}.png")
    plt.show()


if __name__ == '__main__':
    change_working_dir()
    plot("random_mask-"+last_run())#, baseline=0.0168)