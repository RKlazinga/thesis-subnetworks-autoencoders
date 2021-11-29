import os


def change_working_dir():
    os.chdir(os.path.dirname(os.path.dirname(__file__)))
