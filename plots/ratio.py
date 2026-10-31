"""Модель, точное покрытие, границы и планирование нормального эксперимента."""

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from plots.style import setup, save, size_axis, coverage_errorbar
from utils.intervals import ratio_expected_bounds


def ratio_theory_plots(directory="images/variance_ratio"):
    setup()
    paths = {}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    x = np.linspace(-5, 5, 600)
    for variance, label in [(2, "X: дисперсия 2"), (1, "Y: дисперсия 1")]:
        axes[0].plot(x, stats.norm.pdf(x, scale=np.sqrt(variance)), label=label)
    axes[0].set(xlabel="Наблюдение", ylabel="Плотность", title="Нормальные модели, μ₁ = μ₂ = 0")
    axes[0].legend()
    x = np.linspace(0.01, 4, 1000)
    for n in [5, 15, 101]:
        axes[1].plot(x, stats.f.pdf(x, n - 1, n - 1), label=f"n = m = {n}")
    axes[1].axvline(1, color="black", ls="--")
    axes[1].set(xlabel="F = (S²x/S²y)/τ", ylabel="Плотность", title="Распределение точного пивота")
    axes[1].legend()
    paths["model"] = save(fig, directory, "01_model_and_pivot")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for alpha in [0.1, 0.05, 0.01]:
        sizes = np.geomspace(4, 2000, 100).astype(int)
        lengths = [ratio_expected_bounds(n, n, alpha=alpha)[2] for n in sizes]
        axes[0].loglog(sizes, lengths, label=f"Уровень {1-alpha:.0%}")
        grid = np.linspace(0.1, 8, 1000)
        lo, hi = stats.f.ppf([alpha / 2, 1 - alpha / 2], 9, 9)
        axes[1].fill_between(grid, stats.f.pdf(grid, 9, 9), where=(grid >= lo) & (grid <= hi),
                             alpha=0.15, label=f"{1-alpha:.0%}: [{lo:.2f}; {hi:.2f}]")
    axes[0].set(xlabel="n = m", ylabel="E[U − L]", title="Цена повышения доверительного уровня")
    axes[1].plot(grid, stats.f.pdf(grid, 9, 9), color="black")
    axes[1].set(xlabel="Значение пивота", ylabel="Плотность", title="Центральная область F(9, 9)")
    for ax in axes:
        ax.legend()
    paths["confidence"] = save(fig, directory, "09_confidence_levels")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    budget = 200
    ns = np.arange(4, budget - 3)
    lengths = np.array([ratio_expected_bounds(n, budget - n)[2] for n in ns])
    axes[0].plot(ns, lengths)
    optimum = ns[np.argmin(lengths)]
    axes[0].axvline(optimum, color="tab:red", ls="--", label=f"Минимум при n = {optimum}")
    axes[0].set(xlabel="n, при m = 200 − n", ylabel="Ожидаемая длина", ylim=(0, 15),
                title="Фиксированный бюджет наблюдений")
    axes[0].legend()
    grid = np.unique(np.geomspace(4, 300, 30).astype(int))
    values = np.array([[ratio_expected_bounds(n, m)[2] for n in grid] for m in grid])
    heat = axes[1].pcolormesh(grid, grid, np.log10(values), shading="auto", cmap="viridis_r")
    axes[1].set(xscale="log", yscale="log", xlabel="n", ylabel="m",
                title="Длина интервала зависит от обеих выборок")
    fig.colorbar(heat, ax=axes[1], label="log₁₀ E[U − L]")
    paths["allocation"] = save(fig, directory, "11_sample_allocation")
    return paths


def ratio_result_plots(experiment, unbalanced, directory="images/variance_ratio"):
    setup()
    paths = {}
    table, results = experiment["table"], experiment["results"]
    alpha, truth = experiment["alpha"], experiment["truth"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    coverage_errorbar(axes[0], table, "Покрытие ± 95% интервал Уилсона", "tab:blue")
    axes[0].axhline(1-alpha, color="black", ls="--", label="Точное покрытие 0.95")
    size_axis(axes[0], "Доля интервалов, содержащих τ")
    axes[0].set_title("Покрытие не должно расти к единице")
    axes[0].legend(fontsize=9)
    for col, label in [("below", "U < τ"), ("above", "L > τ")]:
        axes[1].plot(table.n, table[col], "o-", label=label)
    axes[1].axhline(alpha/2, color="black", ls="--", label="α/2 = 0.025")
    size_axis(axes[1], "Доля промахов с каждой стороны")
    axes[1].legend()
    paths["coverage"] = save(fig, directory, "02_coverage_and_tails")

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, n in zip(axes, [5, 30, 1000]):
        r = results[n]
        good = (r["low"][:70] <= truth) & (r["high"][:70] >= truth)
        for i in range(70):
            color = "tab:blue" if good[i] else "tab:red"
            ax.plot([r["low"][i], r["high"][i]], [i, i], color=color, alpha=0.7, lw=1.1)
        ax.axvline(truth, color="black", ls="--")
        ax.set(xscale="log", xlabel="Границы интервала (логарифмическая шкала)",
               ylabel="Номер повторения", title=f"n = m = {n}: первые 70 интервалов")
    paths["intervals"] = save(fig, directory, "03_interval_realizations")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    finite = table.n > 3
    t = table[finite]
    axes[0].loglog(t.n, t.mean_length, "o-", label="Средняя длина (эксперимент)")
    axes[0].loglog(t.n, t.theory_length, "--", label="Точное E[U − L]")
    axes[0].loglog(table.n, table.median_length, "s-", label="Медианная длина")
    axes[0].fill_between(table.n, table.length_q10, table.length_q90, alpha=0.15,
                         label="10–90% квантилей длины")
    axes[0].set(xlabel="n = m", ylabel="Длина", title="Убывание длины и тяжёлый правый хвост")
    axes[0].legend(fontsize=9)
    axes[1].plot(t.n, np.sqrt(t.n)*t.mean_length, "o-", label="√n × средняя длина")
    axes[1].plot(t.n, np.sqrt(t.n)*t.theory_length, "--", label="√n × E[длина]")
    axes[1].axhline(4*truth*stats.norm.ppf(1-alpha/2), color="black", ls=":",
                       label="Предел 4τz ≈ 15.68")
    size_axis(axes[1], "Длина × √n")
    axes[1].set_title("Скорость порядка 1/√n")
    axes[1].legend(fontsize=9)
    paths["length"] = save(fig, directory, "04_length_and_rate")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for col, label, color in [("low", "Нижняя граница L", "tab:blue"), ("high", "Верхняя граница U", "tab:orange")]:
        q = np.array([np.quantile(results[n][col], [.1, .5, .9]) for n in table.n])
        axes[0].plot(table.n, q[:, 1], "o-", color=color, label=label)
        axes[0].fill_between(table.n, q[:, 0], q[:, 2], alpha=.15, color=color)
    axes[0].axhline(truth, color="black", ls="--", label="Истинное τ = 2")
    axes[0].set(yscale="log", title="Медианы и 10–90% квантилей границ")
    size_axis(axes[0], "Граница интервала (log)")
    axes[0].legend()
    axes[1].plot(t.n, t.mean_lower, "o-", label="Средняя L")
    axes[1].plot(t.n, t.theory_lower, "--", label="E[L]")
    axes[1].plot(t.n, t.mean_upper, "o-", label="Средняя U")
    axes[1].plot(t.n, t.theory_upper, "--", label="E[U]")
    axes[1].axhline(truth, color="black", ls=":")
    axes[1].set(yscale="log", title="Ожидаемые границы: конечны только при m > 3")
    size_axis(axes[1], "Среднее значение границы (log)")
    axes[1].legend()
    paths["bounds"] = save(fig, directory, "05_boundaries")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    probabilities = (np.arange(1, 500) - .5) / 499
    for ax, n in zip(axes, [5, 30, 1000]):
        theoretical = stats.f.ppf(probabilities, n-1, n-1)
        empirical = np.quantile(results[n]["pivot"], probabilities)
        ax.plot(theoretical, empirical, ".", alpha=.6)
        ax.plot([theoretical[0], theoretical[-1]], [theoretical[0], theoretical[-1]], "k--")
        ax.set(xlabel="Квантили F(n−1, n−1)", ylabel="Выборочные квантили пивота", title=f"Q–Q, n = {n}")
    paths["pivot"] = save(fig, directory, "06_pivot_qq")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, n in zip(axes, [3, 30]):
        lengths = results[n]["high"] - results[n]["low"]
        order = np.sort(lengths)
        ax.loglog(order, (np.arange(len(order), 0, -1))/len(order), label="Эмпирический хвост")
        ax.axvline(np.median(lengths), color="tab:orange", ls="--", label="Медиана длины")
        ax.set(xlabel="Длина l", ylabel="Доля длин ≥ l", title=f"n = m = {n}: " + ("E[длина] = ∞" if n == 3 else "конечное E[длины]"))
        ax.legend()
    paths["heavy_tail"] = save(fig, directory, "07_length_tails")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for n in [30, 100, 1000]:
        r = results[n]
        standardized = np.sqrt(n)/2 * np.log(r["ratio"]/truth)
        axes[0].hist(standardized, bins=80, density=True, histtype="step", label=f"n = {n}")
        q_low, q_high = stats.f.ppf([alpha/2, 1-alpha/2], n-1, n-1)
        axes[1].plot(n, np.sqrt(n)/2*np.log(q_high), "bo")
        axes[1].plot(n, np.sqrt(n)/2*np.log(q_low), "ro")
    x = np.linspace(-4, 4, 500)
    axes[0].plot(x, stats.norm.pdf(x), "k--", label="N(0, 1)")
    axes[0].set(xlabel="(√n / 2) log(R / τ)", ylabel="Плотность", title="Асимптотика логарифма отношения")
    axes[0].legend()
    axes[1].axhline(stats.norm.ppf(1-alpha/2), color="blue", ls="--", label="z₀.₉₇₅")
    axes[1].axhline(stats.norm.ppf(alpha/2), color="red", ls="--", label="z₀.₀₂₅")
    size_axis(axes[1], "Нормированные логарифмы квантилей F")
    axes[1].set_title("Почему длина убывает как 1/√n")
    axes[1].legend()
    paths["log_ratio"] = save(fig, directory, "08_log_ratio_limit")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    labels = [f"({n}, {m})" for n, m in zip(unbalanced.n, unbalanced.m)]
    x = np.arange(len(labels))
    axes[0].errorbar(x, unbalanced.coverage,
                     yerr=[unbalanced.coverage-unbalanced.mc_low, unbalanced.mc_high-unbalanced.coverage],
                     fmt="o", capsize=3)
    axes[0].axhline(1-alpha, color="black", ls="--")
    axes[0].set(ylabel="Покрытие", title="Неравные выборки: точность покрытия сохраняется")
    axes[1].bar(x, unbalanced.mean_length, label="Эксперимент", alpha=.7)
    axes[1].plot(x, unbalanced.theory_length, "ks", label="Точная ожидаемая длина")
    axes[1].set(ylabel="Средняя длина", yscale="log", title="Малая выборка ограничивает точность")
    axes[1].legend()
    for ax in axes:
        ax.set_xticks(x, labels, rotation=40)
        ax.set_xlabel("Пара объёмов (n, m)")
    paths["unbalanced"] = save(fig, directory, "10_unbalanced_samples")
    return paths
