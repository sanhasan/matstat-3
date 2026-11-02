import numpy as np
import pandas as pd
from scipy import stats

from utils.intervals import (
    laplace_intervals, laplace_mean_coverage, laplace_median_coverage,
)


def wilson_interval(successes, repeats, level=0.95):
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


def run_laplace_experiment(sizes, repeats=20000, seed=24422444, alpha=0.05, mu=2.0):
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
