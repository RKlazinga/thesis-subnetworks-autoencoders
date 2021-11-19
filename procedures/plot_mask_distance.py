import os

import torch

from experiments.progressive_mask_drawing import RATIOS
from procedures.channel_prune import mask_dist

run_id = "94d36ab4b"

files = os.listdir(f"runs/{run_id}")

for r in RATIOS:
    ratio_files = [f for f in files if f.startswith(f"keep-{r}-")]
    ratio_files.sort(key=lambda x: int(x.rstrip(".pth").split("-")[-1]))
    print(f"RATIO={r}")
    for idx_a, a in enumerate(ratio_files):
        a = torch.load(f"runs/{run_id}/{a}")
        for idx_b, b in enumerate(ratio_files):
            if idx_b > idx_a:
                print(f"Epoch {idx_a} vs {idx_b}")
                b = torch.load(f"runs/{run_id}/{b}")
                dists = mask_dist(a, b)
                # normalise based on max hamming distance, and ratio
                total_dist = sum(dists) / sum([torch.numel(x) for x in a]) / min(1, 2*r)
                print(dists)
                print(total_dist)
    print()
