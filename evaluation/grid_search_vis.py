import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from evaluation.analyse_latent_weights import plot_latent_count_over_time
from utils.file import change_working_dir

if __name__ == '__main__':
    change_working_dir()

    count_res = defaultdict(list)
    loss_res = defaultdict(list)

    for g in tqdm(os.listdir("grid_search")):
        latent_s, lr = [float(x) for x in g.split("]")[0].removeprefix("[").split(",")]
        counts, losses = plot_latent_count_over_time(g, show=False)
        count, loss = counts[-1], losses[-1]
        count_res[f"latent_sparsity={str(latent_s).ljust(10)}"].append(count)
        loss_res[f"latent_sparsity={str(latent_s).ljust(10)}"].append(loss)
        count_res[f"learning rate={str(lr).ljust(11)}"].append(count)
        loss_res[f"learning rate={str(lr).ljust(11)}"].append(count)

    count_arr = []
    loss_arr = []
    xlabels = []
    ylabels = []
    for k, v in sorted(list(count_res.items()), key=lambda kv: kv[0][:5] + str(format(float(kv[0].split("=")[1]), ".10f"))):
        print(k, v, "avg", str(round(sum(v)/len(v), 2)).ljust(4), "zero count", len([x for x in v if x==0]))
        if k.startswith("latent"):
            ylabels.append(k.split("=")[1].strip())
            count_arr.append(v)
            loss_arr.append(loss_res[k])
        else:
            xlabels.append(k.split("=")[1].strip())

    plt.matshow(np.array(count_arr))
    plt.colorbar()
    plt.gca().set_xticklabels([""] + xlabels)
    plt.xlabel("Learning Rate")
    plt.gca().set_yticklabels([""] + ylabels)
    plt.ylabel("Regularisation")
    plt.show()

    plt.matshow(np.array(loss_arr))
    plt.colorbar()
    plt.gca().set_xticklabels([""] + xlabels)
    plt.xlabel("Learning Rate")
    plt.gca().set_yticklabels([""] + ylabels)
    plt.ylabel("Regularisation")
    plt.show()





