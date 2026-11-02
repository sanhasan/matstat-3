import matplotlib.pyplot as plt
import numpy as np
from scipy import special, stats

from plots.style import COLORS, LABELS, coverage_errorbar, save, setup, size_axis
from utils.intervals import laplace_mean_coverage, laplace_median_coverage


def _laplace_standardized_mean_pdf(t, n):
    t = np.asarray(t, dtype=float)
    scaled_abs = np.sqrt(2 * n) * np.abs(t)
    order = n - .5
    density = np.empty_like(scaled_abs)
    at_zero = scaled_abs == 0
    density[at_zero] = np.sqrt(2 * n) * np.exp(
        special.gammaln(n - .5) - np.log(2) - .5 * np.log(np.pi)
        - special.gammaln(n)
    )
    away_from_zero = scaled_abs[~at_zero]
    density[~at_zero] = np.sqrt(2 * n) * np.exp(
        order * np.log(away_from_zero)
        + np.log(special.kve(order, away_from_zero))
        - away_from_zero
        - order * np.log(2)
        - .5 * np.log(np.pi)
        - special.gammaln(n)
    )
    return density


def laplace_theory_plots(directory="images/laplace_location"):
    setup()
    paths = {}
    x = np.linspace(-4, 8, 800)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(x, stats.laplace.pdf(x, loc=2), label="Laplace(μ=2, b=1)")
    axes[0].plot(x, stats.norm.pdf(x, loc=2, scale=np.sqrt(2)), "--", label="Нормальная с той же дисперсией")
    axes[0].set(xlabel="x", ylabel="Плотность", title="Острый центр и экспоненциальные хвосты")
    axes[1].plot(x, stats.laplace.cdf(x, loc=2))
    axes[1].axvline(2, color="black", ls="--")
    axes[1].axhline(.5, color="tab:orange", ls=":", label="F(μ) = 1/2")
    axes[1].set(xlabel="x", ylabel="F(x)", title="Параметр сдвига — среднее и медиана")
    for ax in axes:
        ax.legend()
    paths["model"] = save(fig, directory, "01_laplace_model")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    t = np.linspace(-4, 4, 600)
    for ax, n in zip(axes, [3, 11, 101]):
        ax.plot(t, _laplace_standardized_mean_pdf(t, n),
                color=COLORS["mean"], label="Точная плотность")
        ax.plot(t, stats.norm.pdf(t), "k--", label="Предел N(0, 1)")
        ax.set(xlabel="√n (среднее − μ) / (√2 b)", ylabel="Плотность",
               title=f"n = {n}", xlim=(-4, 4))
        ax.legend(fontsize=9)
    paths["mean_limit"] = save(fig, directory, "02_exact_mean_density")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, n in zip(axes, [3, 11, 101]):
        t = np.linspace(-4, 4, 600)
        observations = 2 + t/np.sqrt(n)
        cdf = stats.laplace.cdf(observations, loc=2)
        k = (n+1)//2
        log_density = (special.gammaln(n+1) - special.gammaln(k) - special.gammaln(n-k+1)
                       + (k-1)*np.log(cdf) + (n-k)*np.log1p(-cdf)
                       + stats.laplace.logpdf(observations, loc=2) - .5*np.log(n))
        ax.plot(t, np.exp(log_density), color=COLORS["median"], label="Точная плотность")
        ax.plot(t, stats.norm.pdf(t), "k--", label="Предел N(0, 1)")
        ax.set(xlabel="√n (медиана − μ) / b", ylabel="Плотность",
               title=f"n = {n}", xlim=(-4, 4))
        ax.legend(fontsize=9)
    paths["median_limit"] = save(fig, directory, "03_exact_median_density")

    sizes = np.array([3, 5, 11, 21, 51, 101, 301, 1001])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for alpha in [.1, .05, .01]:
        for method, function, style in [("mean", laplace_mean_coverage, "-"),
                                         ("median", laplace_median_coverage, "--")]:
            exact = np.array([function(int(n), alpha) for n in sizes])
            axes[0].plot(sizes, exact-(1-alpha), style, marker="o",
                         label=f"{LABELS[method]}, {1-alpha:.0%}")
        z = stats.norm.ppf(1-alpha/2)
        axes[1].loglog(sizes, 2*z/np.sqrt(sizes), label=f"Медиана, {1-alpha:.0%}")
    axes[0].axhline(0, color="black", ls=":")
    size_axis(axes[0], "Точное покрытие − номинальный уровень")
    axes[0].set_title("Приближение зависит от n и уровня доверия")
    axes[0].legend(fontsize=8, ncols=2)
    axes[1].set(xlabel="n", ylabel="Длина", title="Более высокий уровень даёт большую длину")
    axes[1].legend()
    paths["confidence"] = save(fig, directory, "12_confidence_sensitivity")

    return paths


def laplace_result_plots(experiment, paired, directory="images/laplace_location"):
    setup()
    paths = {}
    table, results = experiment["table"], experiment["results"]
    truth, alpha = experiment["truth"], experiment["alpha"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for method in ["mean", "median"]:
        t = table[table.method == method]
        coverage_errorbar(axes[0], t, LABELS[method]+", Монте-Карло", COLORS[method])
        axes[0].plot(t.n, t.exact_coverage, color=COLORS[method], label=LABELS[method]+", точное покрытие")
        axes[1].plot(t.n, t.coverage-(1-alpha), "o-", color=COLORS[method], label=LABELS[method])
    axes[0].axhline(1-alpha, color="black", ls="--", label="Номинальный уровень 0.95")
    axes[1].axhline(0, color="black", ls="--")
    size_axis(axes[0], "Вероятность покрытия")
    size_axis(axes[1], "Эмпирическое покрытие − 0.95")
    axes[0].set_title("Покрытие асимптотических интервалов")
    axes[1].set_title("Ошибка приближения и случайные колебания")
    for ax in axes:
        ax.legend(fontsize=9)
    paths["coverage"] = save(fig, directory, "03_coverage")

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for col, n in enumerate([5, 51, 1001]):
        for row, method in enumerate(["mean", "median"]):
            ax = axes[row, col]
            r = results[n][method]
            for i in range(60):
                good = r["low"][i] <= truth <= r["high"][i]
                ax.plot([r["low"][i], r["high"][i]], [i, i],
                        color=COLORS[method] if good else "tab:red", alpha=.7, lw=1.2)
            ax.axvline(truth, color="black", ls="--")
            ax.set(xlabel="Границы интервала", ylabel="Номер повторения",
                   title=f"{LABELS[method]}, n = {n}")
        # Одинаковая шкала для двух методов при одном n.
        limits = [ax.get_xlim() for ax in axes[:, col]]
        for ax in axes[:, col]:
            ax.set_xlim(min(lim[0] for lim in limits), max(lim[1] for lim in limits))
    paths["intervals"] = save(fig, directory, "04_paired_interval_realizations")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for method in ["mean", "median"]:
        t = table[table.method == method]
        axes[0].loglog(t.n, t.mean_length, "o-", color=COLORS[method], label=LABELS[method])
        axes[1].plot(t.n, np.sqrt(t.n)*t.mean_length, "o-", color=COLORS[method], label=LABELS[method])
    axes[0].set(xlabel="n", ylabel="Длина интервала", title="Медианный интервал в √2 раз короче")
    size_axis(axes[1], "Длина × √n")
    axes[1].set_title("Точное масштабирование построенных интервалов")
    for ax in axes:
        ax.legend()
    paths["length"] = save(fig, directory, "05_length_comparison")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for method in ["mean", "median"]:
        t = table[table.method == method]
        axes[0].plot(t.n, t.n*t.variance, "o-", color=COLORS[method], label=LABELS[method])
        axes[1].loglog(t.n, t.mse, "o-", color=COLORS[method], label=LABELS[method])
        axes[1].loglog(t.n, t.asymptotic_variance, "--", color=COLORS[method], alpha=.8)
    axes[0].axhline(2, color=COLORS["mean"], ls="--", label="n Var(среднего) = 2")
    axes[0].axhline(1, color=COLORS["median"], ls="--", label="Предел для медианы = 1")
    size_axis(axes[0], "n × эмпирическая дисперсия")
    axes[0].set_title("Асимптотическая эффективность медианы")
    axes[1].set(xlabel="n", ylabel="MSE", title="Пунктир: 2/n и 1/n")
    for ax in axes:
        ax.legend(fontsize=9)
    paths["variance"] = save(fig, directory, "06_variance_and_mse")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, n in zip(axes, [5, 51, 1001]):
        for method in ["mean", "median"]:
            standardized = np.sqrt(n)*(results[n][method]["center"]-truth) / (np.sqrt(2) if method == "mean" else 1)
            ax.hist(standardized, bins=90, density=True, histtype="step", color=COLORS[method], label=LABELS[method])
        x = np.linspace(-4, 4, 600)
        ax.plot(x, stats.norm.pdf(x), "k--", label="N(0, 1)")
        ax.set(xlabel="Стандартизованная ошибка", ylabel="Плотность", xlim=(-4, 4), title=f"n = {n}")
        ax.legend(fontsize=9)
    paths["distribution"] = save(fig, directory, "07_standardized_errors")

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    probabilities = np.linspace(.002, .998, 499)
    theoretical = stats.norm.ppf(probabilities)
    for col, n in enumerate([5, 51, 1001]):
        for row, method in enumerate(["mean", "median"]):
            standardized = np.sqrt(n)*(results[n][method]["center"]-truth) / (np.sqrt(2) if method == "mean" else 1)
            ax = axes[row, col]
            ax.plot(theoretical, np.quantile(standardized, probabilities), ".", color=COLORS[method], alpha=.6)
            ax.plot(theoretical, theoretical, "k--")
            ax.set(xlabel="Квантили N(0, 1)", ylabel="Эмпирические квантили", title=f"{LABELS[method]}, n = {n}")
    paths["qq"] = save(fig, directory, "08_normal_qq")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for method in ["mean", "median"]:
        t = table[table.method == method]
        axes[0].plot(t.n, t.below, "o-", color=COLORS[method], label=LABELS[method]+": U < μ")
        axes[0].plot(t.n, t.above, "s--", color=COLORS[method], label=LABELS[method]+": L > μ")
        axes[1].plot(t.n, t.bias, "o-", color=COLORS[method], label=LABELS[method])
        se = np.sqrt(t.variance/experiment["repeats"])
        axes[1].fill_between(t.n, -1.96*se, 1.96*se, alpha=.12, color=COLORS[method])
    axes[0].axhline(alpha/2, color="black", ls=":")
    axes[1].axhline(0, color="black", ls="--")
    size_axis(axes[0], "Доля промахов")
    size_axis(axes[1], "Эмпирическое смещение")
    axes[0].set_title("Симметрия: два хвоста имеют равные вероятности")
    axes[1].set_title("Полосы: ±1.96 стандартной ошибки среднего оценки")
    for ax in axes:
        ax.legend(fontsize=9)
    paths["tails_bias"] = save(fig, directory, "09_tails_and_bias")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for method in ["mean", "median"]:
        bounds = np.array([np.quantile(results[n][method]["low"], [.1, .5, .9]) for n in results])
        uppers = np.array([np.quantile(results[n][method]["high"], [.1, .5, .9]) for n in results])
        n_values = np.array(list(results))
        ax = axes[0 if method == "mean" else 1]
        ax.plot(n_values, bounds[:, 1], "o-", label="Медиана L")
        ax.fill_between(n_values, bounds[:, 0], bounds[:, 2], alpha=.15)
        ax.plot(n_values, uppers[:, 1], "o-", label="Медиана U")
        ax.fill_between(n_values, uppers[:, 0], uppers[:, 2], alpha=.15)
        ax.axhline(truth, color="black", ls="--")
        size_axis(ax, "Границы: медианы и диапазон 10–90%")
        ax.set_title(LABELS[method]+": границы стягиваются к μ")
        ax.legend()
    paths["bounds"] = save(fig, directory, "10_boundaries")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].errorbar(paired.n, paired.difference, yerr=1.96*paired.paired_se, fmt="o-", capsize=3,
                     label="median − mean, ±1.96 парной SE")
    exact = table.pivot(index="n", columns="method", values="exact_coverage")
    axes[0].plot(exact.index, exact["median"]-exact["mean"], "k--", label="Точная разность покрытий")
    axes[0].axhline(0, color="black", ls=":")
    size_axis(axes[0], "Разность покрытий")
    axes[0].legend(fontsize=9)
    n = 51
    a = results[n]["mean"]["center"]-truth
    b = results[n]["median"]["center"]-truth
    axes[1].scatter(a[:1500], b[:1500], s=8, alpha=.2)
    axes[1].axhline(0, color="black", lw=.8)
    axes[1].axvline(0, color="black", lw=.8)
    axes[1].set(xlabel="Ошибка среднего", ylabel="Ошибка медианы",
                title=f"Одни данные, коррелированные оценки, n = {n}")
    paths["paired"] = save(fig, directory, "11_paired_comparison")
    return paths
