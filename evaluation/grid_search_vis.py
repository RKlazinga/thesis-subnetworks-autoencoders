import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from evaluation.analyse_latent_weights import plot_latent_count_over_time
from utils.file import change_working_dir

if __name__ == '__main__':
    change_working_dir()

    res = defaultdict(list)

    for g in tqdm(os.listdir("grid_search")):
        if "newer" not in g:
            continue
        sparsity, latent_s, lr = [float(x) for x in g.split("]")[0].removeprefix("[").split(",")]
        c = plot_latent_count_over_time(g, show=False)[-1]
        res[f"non_latent_sparsity={str(sparsity).ljust(5)}"].append(c)
        res[f"latent_sparsity={str(latent_s).ljust(10)}"].append(c)
        res[f"learning rate={str(lr).ljust(11)}"].append(c)

    arr = []
    for k, v in sorted(list(res.items()), key=lambda kv: kv[0][:5]+str(format(float(kv[0].split("=")[1]), ".10f"))):
        print(k, v, "avg", str(round(sum(v)/len(v), 2)).ljust(4), "zero count", len([x for x in v if x==0]))
        if k.startswith("latent"):
            arr.append(v)

    plt.matshow(np.array(arr))
    plt.colorbar()
    plt.show()





