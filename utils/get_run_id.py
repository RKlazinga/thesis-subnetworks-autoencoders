import os

from settings.global_settings import RUN_FOLDER
from utils.file import change_working_dir


def last_group():
    change_working_dir()
    groups = [x for x in os.listdir(RUN_FOLDER) if os.path.isdir(os.path.join(RUN_FOLDER, x)) and "GROUP" in x.split("-")[0]]
    groups.sort(key=lambda x: os.path.getmtime(os.path.join(RUN_FOLDER, x)), reverse=True)
    if len(groups) == 0:
        raise IndexError("Could not get last group: no groups saved")
    return groups[0]


def _sort_runs_by_time():
    change_working_dir()
    runs = [x for x in os.listdir(RUN_FOLDER) if os.path.isdir(os.path.join(RUN_FOLDER, x)) and "GROUP" not in x.split("-")[0]]
    runs.sort(key=lambda x: os.path.getmtime(os.path.join(RUN_FOLDER, x)), reverse=True)
    if len(runs) == 0:
        raise IndexError("Could not get last run: no runs saved")
    return runs


def last_runs(count, offset=0):
    runs = _sort_runs_by_time()
    return runs[offset:count+offset]


def last_run():
    return _sort_runs_by_time()[0]


def all_runs_matching(prefix):
    change_working_dir()
    runs = [x for x in os.listdir(RUN_FOLDER) if os.path.isdir(os.path.join(RUN_FOLDER, x))]
    return [x for x in runs if x.startswith(prefix)]
