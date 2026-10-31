"""Воспроизводимое моделирование с ограниченным расходом памяти."""

import numpy as np
import pandas as pd
from scipy import stats

from utils.intervals import (
    laplace_intervals, laplace_mean_coverage, laplace_median_coverage,
    ratio_expected_bounds, ratio_from_samples, variance_ratio_interval,
)


def wilson_interval(successes, repeats, level=0.95):
    """Интервал Уилсона для вероятности покрытия в эксперименте Монте-Карло."""
    p = np.asarray(successes) / repeats
    z = stats.norm.ppf((1 + level) / 2)
    denom = 1 + z**2 / repeats
    center = (p + z**2 / (2 * repeats)) / denom
    half = z * np.sqrt(p * (1 - p) / repeats + z**2 / (4 * repeats**2)) / denom
    return center - half, center + half


def summarize_intervals(low, high, truth):
    low, high = np.asarray(low), np.asarray(high)
    covered = (low <= truth) & (truth <= high)
    ci_low, ci_high = wilson_interval(covered.sum(), covered.size)
    length = high - low
    return {
        "coverage": covered.mean(), "mc_se": covered.std(ddof=1) / np.sqrt(covered.size),
        "mc_low": float(ci_low), "mc_high": float(ci_high),
        "below": (high < truth).mean(), "above": (low > truth).mean(),
        "mean_lower": low.mean(), "mean_upper": high.mean(),
        "mean_length": length.mean(), "median_length": np.median(length),
        "length_q10": np.quantile(length, 0.1), "length_q90": np.quantile(length, 0.9),
    }


def run_ratio_experiment(sizes, repeats=20000, seed=24422442, alpha=0.05):
    """Основной эксперимент: реальные нормальные выборки, n=m."""
    rng = np.random.default_rng(seed)
    results, rows = {}, []
    for n in map(int, sizes):
        # Не более примерно 16 MiB на одну из двух матриц наблюдений.
        batch = max(1, min(512, 2_000_000 // n))
        lows, highs, ratios = [], [], []
        for start in range(0, repeats, batch):
            count = min(batch, repeats - start)
            x = rng.normal(0, np.sqrt(2), size=(count, n))
            y = rng.normal(0, 1, size=(count, n))
            low, high = ratio_from_samples(x, y, alpha)
            lows.append(low)
            highs.append(high)
            ratios.append(x.var(axis=-1, ddof=1) / y.var(axis=-1, ddof=1))
        low, high, ratio = map(np.concatenate, (lows, highs, ratios))
        results[n] = {"low": low, "high": high, "ratio": ratio, "pivot": ratio / 2}
        expected_low, expected_high, expected_length = ratio_expected_bounds(n, n, alpha=alpha)
        rows.append({"n": n, "m": n, **summarize_intervals(low, high, 2),
                     "theory_lower": expected_low, "theory_upper": expected_high,
                     "theory_length": expected_length})
    return {"results": results, "table": pd.DataFrame(rows), "alpha": alpha,
            "repeats": repeats, "truth": 2.0}


def run_unbalanced_experiment(pairs, repeats=40000, seed=24422443, alpha=0.05):
    """Дополнение: точная генерация достаточных статистик через хи-квадрат."""
    rng = np.random.default_rng(seed)
    rows = []
    for n, m in pairs:
        var_x = 2 * rng.chisquare(n - 1, repeats) / (n - 1)
        var_y = rng.chisquare(m - 1, repeats) / (m - 1)
        low, high = variance_ratio_interval(var_x, var_y, n, m, alpha)
        rows.append({"n": n, "m": m, **summarize_intervals(low, high, 2),
                     "theory_length": ratio_expected_bounds(n, m, alpha=alpha)[2]})
    return pd.DataFrame(rows)


def run_laplace_experiment(sizes, repeats=20000, seed=24422444, alpha=0.05, mu=2.0):
    """Оба метода на одних и тех же выборках: сравнение парное."""
    rng = np.random.default_rng(seed)
    results, rows = {}, []
    for n in map(int, sizes):
        centers = {"mean": [], "median": []}
        intervals = {method: [[], []] for method in centers}
        batch = max(1, min(512, 2_000_000 // n))
        for start in range(0, repeats, batch):
            sample = rng.laplace(mu, 1, size=(min(batch, repeats - start), n))
            current = laplace_intervals(sample, alpha)
            for method, (low, high) in current.items():
                intervals[method][0].append(low)
                intervals[method][1].append(high)
                centers[method].append((low + high) / 2)
        results[n] = {}
        for method in centers:
            low, high = map(np.concatenate, intervals[method])
            center = np.concatenate(centers[method])
            results[n][method] = {"low": low, "high": high, "center": center}
            exact_coverage = (laplace_mean_coverage(n, alpha) if method == "mean"
                              else laplace_median_coverage(n, alpha) if n % 2 else np.nan)
            rows.append({"n": n, "method": method,
                         **summarize_intervals(low, high, mu),
                         "exact_coverage": exact_coverage,
                         "bias": (center - mu).mean(),
                         "variance": center.var(ddof=1),
                         "mse": np.mean((center - mu)**2),
                         "asymptotic_variance": (2 if method == "mean" else 1) / n})
    return {"results": results, "table": pd.DataFrame(rows), "alpha": alpha,
            "repeats": repeats, "truth": mu}


def paired_coverage_difference(experiment):
    """Разность покрытий median - mean и её парная стандартная ошибка."""
    rows = []
    truth = experiment["truth"]
    for n, current in experiment["results"].items():
        indicators = {}
        for method, result in current.items():
            indicators[method] = ((result["low"] <= truth) & (truth <= result["high"])).astype(float)
        delta = indicators["median"] - indicators["mean"]
        rows.append({"n": n, "difference": delta.mean(),
                     "paired_se": delta.std(ddof=1) / np.sqrt(delta.size)})
    return pd.DataFrame(rows)
