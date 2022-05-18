from enum import Enum

import numpy as np
import skdim
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.dataset_options import DatasetOption
from datasets.synthetic.flat_generators import random_sine_gaussian
from datasets.synthetic.synthetic_flat import SyntheticFlat
from datasets.synthetic.synthetic_im import Synthetic
from evaluation.analyse_latent_weights import analyse_at_epoch, plot_latent_count_over_time
from experiments.progressive_mask_drawing import train_and_draw_tickets
from models.ff_ae import FeedforwardAE
from settings import Settings
from utils.misc import dev, generate_random_str


class Benchmarks(Enum):
    SINE = [None, 16, 2]
    SWISS_ROLL = ("M7_Roll", 3, 2)
    PARABOLOID3 = ("Mp1_Paraboloid", 12, 3)
    CONCENTRATED = ("M3_Nonlinear_4to6", 6, 4)
    MANIFOLD = ("M4_Nonlinear", 8, 4)
    HYPERCUBE = ("M10a_Cubic", 11, 10)
    AFFINE20 = ("M9_Affine", 20, 20)
    HELIX = ("M13b_Spiral", 13, 1)
    MOEBIUS = ("M11_Moebius", 3, 2)
    HYPERSPHERE = ("M1_Sphere", 11, 10)
    MANIFOLD6 = ("M6_Nonlinear", 36, 6)
    PARABOLOID6 = ("Mp2_Paraboloid", 21, 6)
    AFFINE = ("M2_Affine_3to5", 5, 3)


def our_id_est(benchmark: Benchmarks):
    Settings.NUM_VARIABLES = benchmark.value[2]
    Settings.DS = DatasetOption.SYNTHETIC_FLAT

    Settings.TRAIN_SIZE = 5000
    Settings.DRAW_EPOCHS = int(400000 / Settings.TRAIN_SIZE)

    if benchmark != Benchmarks.SINE:
        Settings.OVERRIDE_SYNTH_GENERATOR = lambda count: skdim.datasets\
            .BenchmarkManifolds()\
            .generate(benchmark.value[0], count, *benchmark.value[1:])
        Settings.FLAT_DATAPOINTS = benchmark.value[1]

    # best known settings
    Settings.LATENT_SPARSITY_PENALTY = 1e-1
    Settings.LINEAR_SPARSITY_PENALTY = 1e-3

    Settings.TOPOLOGY = [Settings.FLAT_DATAPOINTS, 5, 1]
    network = Settings.NETWORK(*Settings.TOPOLOGY).to(dev())
    run_id = f"comparative-{generate_random_str()}"
    dataset: SyntheticFlat = train_and_draw_tickets(network, run_id)

    # build a single tensor from the points in the dataset
    combined_data = torch.stack(dataset.imgs, dim=0)

    # get our id estimate
    return plot_latent_count_over_time(run_id, show=False)[0][-1], combined_data
    # weights = analyse_at_epoch(run_id, Settings.DRAW_EPOCHS)[0]
    # return len([w for w in weights if abs(w) > 2e-4]), combined_data


if __name__ == '__main__':
    b = Benchmarks.CONCENTRATED
    if b == Benchmarks.SINE:
        b.value[2] = 4
    Settings.NORMAL_STD_DEV = 0.1
    our_estimate, data = our_id_est(b)
    other_methods = [
        skdim.id.CorrInt(),
        skdim.id.FisherS(),
        skdim.id.lPCA(),
        skdim.id.MADA(),
        skdim.id.MLE(),
        skdim.id.MOM(),
        skdim.id.TLE(),
        skdim.id.TwoNN()
    ]
    print("=" * 50)
    print()
    print("Alignment:")
    print("l" + "c" * (len(other_methods) + 1))
    print("First table row:")
    print(f"Dataset & True ID & Ours & {' & '.join(x.__class__.__name__ for x in other_methods)} \\\\")
    true_id = b.value[2]
    print(b.name.replace("_", " ").lower().title(), end=" & ")
    print(true_id, end=" & ")
    if our_estimate == true_id:
        print(f"\\textbf{{{our_estimate}}}", end=" & ")
    else:
        print(our_estimate, end=" & ")
    for o in other_methods:
        res = round(o.fit(data).dimension_, 2)
        correct = abs(true_id - res) < 0.5
        if res != int(res):
            res = f"{res:.2f}"
        if correct:
            print(f"\\textbf{{{res}}}", end="")
        else:
            print(res, end="")

        if o != other_methods[-1]:
            print("&", end="")
        else:
            print(r"\\")
