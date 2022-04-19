import os

from settings.s import Settings
from utils.file import change_working_dir


def last_group():
    change_working_dir()
    groups = [x for x in os.listdir(Settings.RUN_FOLDER) if os.path.isdir(os.path.join(Settings.RUN_FOLDER, x)) and "GROUP" in x.split("-")[0]]
    groups.sort(key=lambda x: os.path.getmtime(os.path.join(Settings.RUN_FOLDER, x)), reverse=True)
    if len(groups) == 0:
        raise IndexError("Could not get last group: no groups saved")
    return groups[0]


def _sort_runs_by_time():
    change_working_dir()
    runs = [x for x in os.listdir(Settings.RUN_FOLDER) if os.path.isdir(os.path.join(Settings.RUN_FOLDER, x)) and "GROUP" not in x.split("-")[0]]
    runs.sort(key=lambda x: os.path.getmtime(os.path.join(Settings.RUN_FOLDER, x)), reverse=True)
    if len(runs) == 0:
        raise IndexError("Could not get last run: no runs saved")
    return runs


def last_runs(count, offset=0):
    runs = _sort_runs_by_time()
    return runs[offset:count+offset]


def last_run(offset=0):
    return _sort_runs_by_time()[offset]


def all_runs_matching(prefix_or_term):
    change_working_dir()
    runs = [x for x in os.listdir(Settings.RUN_FOLDER) if os.path.isdir(os.path.join(Settings.RUN_FOLDER, x))]
    matches = [x for x in runs if x.startswith(prefix_or_term)]
    if len(matches) > 0:
        return matches
    return [x for x in runs if prefix_or_term in x]
