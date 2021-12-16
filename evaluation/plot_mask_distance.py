import os

import torch
import seaborn as sns
import matplotlib.pyplot as plt

from procedures.ticket_drawing.without_redist import mask_dist
from settings.prune_settings import *
from utils.file import change_working_dir
from utils.get_run_id import last_run

change_working_dir()

run_id = last_run()
files = os.listdir(f"runs/{run_id}")

FIG_COUNT = 4
ROW_SIZE = 2

fig, axs = plt.subplots(FIG_COUNT // ROW_SIZE, ROW_SIZE)
fig.tight_layout()
PRUNE_RATIOS.sort()

for idx_r, r in enumerate(PRUNE_RATIOS):
    ratio_files = [f for f in files if f.startswith(f"keep-{r}-")]

    def get_epoch_and_iter(x: str):
        x = x.removesuffix(".pth")
        epoch = int(x.split("-")[-2])
        iteration = int(x.split("-")[-1])
        return epoch + iteration / DRAW_PER_EPOCH

    ratio_files.sort(key=get_epoch_and_iter)

    dists = torch.zeros((len(ratio_files), len(ratio_files)), dtype=torch.float)

    for idx_a, a in enumerate(ratio_files):
        a = torch.load(f"runs/{run_id}/{a}", map_location=torch.device('cpu'))
        for idx_b, b in enumerate(ratio_files):
            if idx_b > idx_a:
                b = torch.load(f"runs/{run_id}/{b}", map_location=torch.device('cpu'))
                dist = mask_dist(a, b)
                # normalise based on max hamming distance, and ratio
                total_dist = sum(dist) / sum([torch.numel(x) for x in a]) / min(1, 2*r)
                dists[idx_a][idx_b] = total_dist
                dists[idx_b][idx_a] = total_dist

    ax = axs[idx_r // ROW_SIZE, idx_r % ROW_SIZE]
    ax.set_title(f"Ratio {r}")

    xticklabels = [x//DRAW_PER_EPOCH if x % DRAW_PER_EPOCH == 0 else None for x in range(len(ratio_files))]
    sns.heatmap(dists, ax=ax, square=True, xticklabels=xticklabels, yticklabels=False)
plt.show()
