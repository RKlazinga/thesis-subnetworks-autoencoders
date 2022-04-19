import os

import torch.cuda

from settings.s import Settings
from utils.misc import dev


def get_topology_of_run(run_id):
    change_working_dir()
    checkpoint_file = [x for x in os.listdir(f"{Settings.RUN_FOLDER}/{run_id}") if x.startswith("starting_params-")][0]
    checkpoint_settings = checkpoint_file.removesuffix("].pth").removeprefix("starting_params-[")
    checkpoint_settings = [int(x) for x in checkpoint_settings.split(",")]
    return checkpoint_settings


def get_params_of_run(run_id, epoch=None, device=None):
    if device is None:
        device = dev()

    if epoch is None:
        checkpoint_file = [x for x in os.listdir(f"{Settings.RUN_FOLDER}/{run_id}") if x.startswith("starting_params-")][0]
    else:
        checkpoint_file = f"trained-{epoch}.pth"
    checkpoint_file = f"{Settings.RUN_FOLDER}/{run_id}/" + checkpoint_file
    return torch.load(checkpoint_file, map_location=device)


def get_epochs_of_run(run_id):
    change_working_dir()
    files = [x for x in os.listdir(f"{Settings.RUN_FOLDER}/{run_id}") if "trained" in x]
    files = [int(x.removesuffix(".pth").removeprefix("trained-")) for x in files]
    return max(files)


def get_all_current_settings():
    change_working_dir()
    settings_files = os.listdir("settings")
    out = ""
    for s in settings_files:
        out += "=== " + s + "\n"
        if os.path.isfile(f"settings/{s}") and "__"  not in s:
            with open(f"settings/{s}", "r") as readfile:
                out += readfile.read()
        out += "\n\n"
    return out


def change_working_dir():
    os.chdir(os.path.dirname(os.path.dirname(__file__)))
