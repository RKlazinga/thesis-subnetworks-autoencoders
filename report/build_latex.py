import math
import os
import shutil
import textwrap
import pyperclip
from tqdm import tqdm

from evaluation.analyse_latent_weights import plot_latent_count_over_time
from utils.file import change_working_dir
from utils.get_run_id import all_runs_matching

FIGURE = textwrap.dedent(r"""
\begin{FIGTYPE}[h!]
\centering
SUBFIG
\caption{CAPTION}
\label{fig:LABEL}
\end{FIGTYPE}
""")
SUBFIG = textwrap.dedent(r"""\begin{subfigure}[b]{WIDTH\textwidth}
    \centering
    \includegraphics[width=\textwidth]{"FNAME"}
    \caption{CAPTION}
    \label{subfig:LABEL}
\end{subfigure}
""")


def table_of_grid_search(match_term=None):
    dims = [1, 2, 3, 4]
    lats = [8]
    for d in dims:
        for l_idx, l in enumerate(lats):
            if l_idx == 0:
                if d > 1:
                    print("\cline{0-1}", end="")
                print(f"\multirow{{{len(lats)}}}{{*}}{{{d}}}", end="")
            print(f" & {l}", end="")
            for latent_sparsity in [1e-3, 1e-2, 1e-1, 1]:
                for linear_sparsity in [1e-3, 1e-2, 1e-1]:
                    runs = all_runs_matching(f"[{l}-{d}-{latent_sparsity}-{linear_sparsity}]-")
                    if match_term:
                        runs = [x for x in runs if match_term in x]
                    assert len(runs) == 1
                    count = plot_latent_count_over_time(runs[0], show=False)[0][-1]
                    if count == d:
                        count = f"\\textbf{{{count}}}"
                    print(f" & {count}", end="")
            print(" \\\\")


def table_of_grid_search_one_col(match_term=None):
    dims = [1, 2, 3, 4]
    lat = 8
    for d in dims:
        print(f"\multirow{{1}}{{*}}{{{d}}}", end="")
        for latent_sparsity in [1e-3, 1e-2, 1e-1, 1]:
            for linear_sparsity in [1e-3, 1e-2, 1e-1]:
                runs = all_runs_matching(f"[{lat}-{d}-{latent_sparsity}-{linear_sparsity}]-")
                if match_term:
                    runs = [x for x in runs if match_term in x]
                assert len(runs) == 1
                count = plot_latent_count_over_time(runs[0], show=False)[0][-1]
                if count == d:
                    count = f"\\textbf{{{count}}}"
                print(f" & {count}", end="")
        print(" \\\\")


def table_of_grid_search_one_col_flipped(match_term=None):
    dims = [1, 2, 3, 4]
    lat = 8
    observed_shots = None
    for latent_sparsity in [1e-3, 1e-2, 1e-1, 1]:
        print(f"\multirow{{1}}{{*}}{{$10^{{{round(math.log10(latent_sparsity))}}}$}}", end="")
        for d in dims:
            for linear_sparsity in [1e-3, 1e-2, 1e-1]:
                runs = all_runs_matching(f"[{lat}-{d}-{latent_sparsity}-{linear_sparsity}]-")
                if match_term:
                    runs = [x for x in runs if match_term in x]
                if observed_shots is None:
                    observed_shots = len(runs)
                    # print("SHOTS:", observed_shots)

                assert len(runs) in [observed_shots, 0]
                if len(runs) == 0:
                    count = -1
                else:
                    # get the result for each shot, then take the median
                    assert observed_shots % 2 == 1, "Shot count must be odd to find median"
                    counts = []
                    for r in runs:
                        single_count = plot_latent_count_over_time(r, show=False)[0][-1]
                        counts.append(single_count)
                    count = sorted(counts)[observed_shots//2]
                    if count == d:
                        count = f"\\textbf{{{count}}}"
                    if len(set(counts)) > 1:
                        count = f"\\textit{{{count}}}"
                print(f" & {count}", end="")
        print(" \\\\")


def table_of_count_freqs(match_term=None):
    dims = [1, 2, 3, 4]
    lat = 8
    observed_shots = None
    cols = []
    for d in dims:
        zero_count = 0
        one_count = 0
        high_count = 0
        for latent_sparsity in tqdm([1e-3, 1e-2, 1e-1, 1]):
            for linear_sparsity in [1e-3, 1e-2, 1e-1]:
                runs = all_runs_matching(f"[{lat}-{d}-{latent_sparsity}-{linear_sparsity}]-")
                if match_term:
                    runs = [x for x in runs if match_term in x]
                if observed_shots is None:
                    observed_shots = len(runs)
                    print("SHOTS:", observed_shots)
                assert len(runs) in [observed_shots, 0]
                if len(runs) != 0:
                    # get the result for each shot, then take the median
                    assert observed_shots % 2 == 1, "Shot count must be odd to find median"
                    for r in runs:
                        single_count = plot_latent_count_over_time(r, show=False)[0][-1]
                        if single_count == 0:
                            zero_count += 1
                        elif single_count == 1:
                            one_count += 1
                        else:
                            high_count += 1
        cols.append((zero_count, one_count, high_count))

    print("0", end="")
    for z, o, h in cols:
        print(" & ", end="")
        print(f"{round(z/(z+o+h)*100)}\\%", end="")
    print(" \\\\")
    print("1", end="")
    for z, o, h in cols:
        print(" & ", end="")
        print(f"{round(o/(z+o+h)*100)}\\%", end="")
    print(" \\\\")
    print("2+", end="")
    for z, o, h in cols:
        print(" & ", end="")
        print(f"{round(h/(z+o+h)*100)}\\%", end="")
    print(" \\\\")


def figure_of_runs(run_ids, plot_type="c", label="", captioner=None, max_row_width=4, max_fig_width=1/3):
    change_working_dir()

    assert plot_type in ["c", "s"]
    if plot_type == "c":
        plot_type = "latent_count"
        caption = "Number of active latent neurons and test loss, over training epochs."
    if plot_type == "s":
        plot_type = "latent_strengths"
        caption = "Channel strength of each latent neuron over training epochs."

    r = len(run_ids)
    if r <= 2:
        figtype = "figure"
    else:
        figtype = "figure*"

    if r <= max_row_width:
        per_row = r
        fig_width = 1/r
    elif (r % 2) == 0 and r <= max_row_width * 2:
        per_row = max_row_width
        fig_width = 1 / (r // 2)
    else:
        row_count = max_row_width
            # math.ceil(run_ids / max_row_width)
        fig_width = 1/max_row_width
    fig_width = round(min(max_fig_width, fig_width), 3) - 0.01

    subfigs = ""
    files = []
    for idx, run_id in enumerate(run_ids):
        if idx > 0 and idx % per_row == 0:
            subfigs += "\n"
        file = f"figures/{plot_type}/{run_id}.png"
        files.append(file)
        txt = SUBFIG
        txt = txt.replace("WIDTH", str(fig_width))
        txt = txt.replace("FNAME", f"fig/{label}/{run_id}")
        if captioner:
            txt = txt.replace("CAPTION", captioner(run_id))
        else:
            txt = txt.replace("CAPTION", "")
        txt = txt.replace("LABEL", f"{label}_{chr(97+idx)}")
        subfigs += txt

    fig = FIGURE
    fig = fig.replace("FIGTYPE", figtype)
    fig = fig.replace("SUBFIG", subfigs)
    fig = fig.replace("CAPTION", caption)
    fig = fig.replace("LABEL", label)

    pyperclip.copy(fig)

    input("Enter to collect files for upload...")
    export_folder = f"export_fig/{label}"
    if os.path.exists(export_folder):
        shutil.rmtree(export_folder)
    os.makedirs(export_folder)
    for f in files:
        shutil.copyfile(f, f"{export_folder}/{os.path.basename(f)}")
    os.startfile(os.path.realpath("export_fig"))

    print(f"Upload the folder '{label}' to 'fig/{label}'")


if __name__ == '__main__':
    table_of_grid_search_one_col_flipped("[SUFFIX_OF_GRID_SEARCH]")