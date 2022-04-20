import os

import torch
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

from procedures.ticket_drawing import mask_dist
from settings import Settings
from utils.file import change_working_dir
from utils.get_run_id import last_run
plt.rcParams["font.family"] = "serif"

change_working_dir()

run_id = last_run()
folder = f"{Settings.RUN_FOLDER}/{run_id}/masks"
files = os.listdir(folder)

FIG_COUNT = 4
ROW_SIZE = 2

fig, axs = plt.subplots(FIG_COUNT // ROW_SIZE, ROW_SIZE)
fig.tight_layout()
Settings.PRUNE_RATIOS.sort(reverse=True)

for idx_r, r in enumerate(Settings.PRUNE_RATIOS):
    ratio_files = [f for f in files if f.startswith(f"prune-{r}-")]

    def get_epoch_and_iter(x: str):
        x = x.removesuffix(".pth")
        epoch = int(x.split("-")[-2])
        iteration = int(x.split("-")[-1])
        return epoch + iteration / Settings.DRAW_PER_EPOCH

    ratio_files.sort(key=get_epoch_and_iter)

    dists = torch.zeros((len(ratio_files), len(ratio_files)), dtype=torch.float)

    for idx_a, a in tqdm(enumerate(ratio_files)):
        a = torch.load(f"{folder}/{a}", map_location=torch.device('cpu'))
        for idx_b, b in enumerate(ratio_files):
            if idx_b > idx_a:
                b = torch.load(f"{folder}/{b}", map_location=torch.device('cpu'))
                dist = mask_dist(a, b)
                # normalise based on max hamming distance, and ratio
                total_dist = sum(dist) / sum([torch.numel(x) for x in a]) / min(1, 2*r)
                dists[idx_a][idx_b] = total_dist
                dists[idx_b][idx_a] = total_dist

    ax = axs[idx_r // ROW_SIZE, idx_r % ROW_SIZE]
    ax.set_title(f"Ratio {r}")

    xticklabels = [x//Settings.DRAW_PER_EPOCH if x % (Settings.DRAW_PER_EPOCH*2) == 0 else None for x in range(len(ratio_files))]
    sns.heatmap(dists, ax=ax, square=True, xticklabels=xticklabels, yticklabels=False)

plt.tight_layout()
plt.savefig(f"figures/mask_distance/{run_id.replace('/', ' X ')}.png")
plt.show()
