import os

from utils.file import change_working_dir


def last_run():
    change_working_dir()
    runs = [x for x in os.listdir("runs") if os.path.isdir(os.path.join("runs", x))]
    runs.sort(key=lambda x: os.path.getmtime(os.path.join("runs", x)), reverse=True)
    if len(runs) == 0:
        raise IndexError("Could not get last run: no runs saved")
    return runs[0]


def all_runs_matching(prefix):
    change_working_dir()
    runs = [x for x in os.listdir("runs") if os.path.isdir(os.path.join("runs", x))]
    return [x for x in runs if x.startswith(prefix)]