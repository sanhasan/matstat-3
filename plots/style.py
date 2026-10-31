"""Единое оформление и сохранение рисунков, как в matstat-2."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = {"mean": "tab:blue", "median": "tab:orange"}
LABELS = {"mean": "Среднее", "median": "Медиана"}


def setup():
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.titlesize": 12, "figure.dpi": 110})


def save(fig, directory, name):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    path = directory / f"{name}.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def size_axis(ax, ylabel):
    ax.set_xscale("log")
    ax.set_xlabel("Объём выборки n")
    ax.set_ylabel(ylabel)


def coverage_errorbar(ax, table, label, color):
    ax.errorbar(table.n, table.coverage,
                yerr=[table.coverage - table.mc_low, table.mc_high - table.coverage],
                fmt="o", capsize=3, label=label, color=color)
