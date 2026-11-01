"""Оформление и вывод графиков по готовым данным из первого ноутбука."""

from plots.style import save, setup

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, ScalarFormatter
import numpy as np
from IPython.display import Image, display


def _show(fig, directory, name):
    path = save(fig, directory, name)
    display(Image(filename=str(path)))


def plot_fisher_quantiles(distribution, q_low, q_high, df, alpha, directory):
    """Создаёт сетки для рисунка и отмечает вычисленные в ноутбуке квантили."""
    setup()
    right_edge = distribution.ppf(1 - alpha / 10)
    grid = np.linspace(0.001, right_edge, 1200)
    density = distribution.pdf(grid)
    central_grid = np.linspace(q_low, q_high, 500)
    central_density = distribution.pdf(central_grid)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(grid, density, color="tab:blue", label=f"Плотность F({df[0]}, {df[1]})")
    ax.fill_between(central_grid, central_density, alpha=0.25, color="tab:blue",
                    label=f"Центральная область: вероятность {1 - alpha:g}")
    ax.axvline(q_low, color="tab:orange", linestyle="--",
               label=f"q({alpha / 2:g}) = {q_low:.3f}")
    ax.axvline(q_high, color="tab:red", linestyle="--",
               label=f"q({1 - alpha / 2:g}) = {q_high:.3f}")
    ax.set(xlabel="Значение R / τ", ylabel="Плотность",
           title="Область, из которой получен доверительный интервал", xlim=(0, grid[-1]))
    ax.legend()
    _show(fig, directory, "task_1_fisher_quantiles")


def plot_coverage(sizes, coverage, level, truth, directory):
    """Рисует готовые доли покрытия и теоретический доверительный уровень."""
    setup()
    half_range = max(0.005, 1.2 * max(abs(value - level) for value in coverage))
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.axhline(level, color="black", linestyle="--", label=f"Теоретическая доля: {level:g}")
    ax.plot(sizes, coverage, "o-", color="tab:blue", label="Доля в эксперименте")
    ax.set(xscale="log", xlabel="Объём каждой выборки n = m",
           ylabel=f"Доля интервалов, содержащих τ = {truth:g}",
           title="Зависимость доли «хороших» интервалов от объёма выборок",
           ylim=(max(0, level - half_range), min(1, level + half_range)))
    ax.set_xticks(sizes, labels=sizes, rotation=35)
    ax.legend(loc="best")
    _show(fig, directory, "task_1_coverage")


def plot_interval_examples(lower, upper, good, truth, n, directory):
    """Рисует готовые интервалы одного объёма, подбирая отдельный масштаб."""
    setup()
    colors = ["tab:blue" if covered else "tab:red" for covered in good]
    positions = range(1, len(lower) + 1)
    left_edge = min(min(lower), truth)
    right_edge = max(max(upper), truth)
    padding = 0.05 * (right_edge - left_edge)
    if padding == 0:
        padding = 0.05 * max(abs(truth), 1)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.hlines(positions, lower, upper, colors=colors, linewidth=1.6)
    ax.scatter(lower, positions, c=colors, marker="|", s=25)
    ax.scatter(upper, positions, c=colors, marker="|", s=25)
    ax.axvline(truth, color="black", linestyle="--", label=f"Истинное τ = {truth:g}")
    ax.plot([], [], color="tab:blue", label="Содержит τ")
    ax.plot([], [], color="tab:red", label="Не содержит τ")
    ax.set(xlabel="Значение отношения дисперсий", ylabel="Номер повторения",
           title=f"Первые {len(lower)} доверительных интервалов, n = m = {n}",
           xlim=(max(0, left_edge - padding), right_edge + padding),
           ylim=(0, len(lower) + 1))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=8))
    formatter = ScalarFormatter(useOffset=False)
    formatter.set_scientific(False)
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=9)
    _show(fig, directory, f"task_1_interval_examples_n_{n}")


def plot_bounds_and_length(sizes, mean_lower, mean_upper, mean_length, truth, directory):
    """Рисует рассчитанные в ноутбуке средние границы и длины."""
    setup()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].plot(sizes, mean_lower, "o-", label="Средняя нижняя граница")
    axes[0].plot(sizes, mean_upper, "o-", label="Средняя верхняя граница")
    axes[0].axhline(truth, color="black", linestyle="--", label=f"Истинное τ = {truth:g}")
    axes[0].set(ylabel="Среднее значение границы", title="Приближение границ к истинному параметру")
    axes[0].legend()
    axes[1].plot(sizes, mean_length, "o-", color="tab:purple")
    axes[1].set(ylabel="Средняя длина U − L", title="Сужение доверительного интервала")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("Объём каждой выборки n = m")
        ax.set_xticks(sizes, labels=sizes, rotation=35)
        ax.set_ylim(bottom=0)
    _show(fig, directory, "task_1_bounds_and_length")
