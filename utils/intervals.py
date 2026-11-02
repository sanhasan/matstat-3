import numpy as np
from scipy import integrate, stats


def validate_alpha(alpha):
    if not 0 < alpha < 1:
        raise ValueError("Требуется 0 < alpha < 1.")


def laplace_intervals(sample, alpha=0.05, scale=1.0):
    validate_alpha(alpha)
    if scale <= 0:
        raise ValueError("Масштаб должен быть положительным.")
    sample = np.asarray(sample)
    n = sample.shape[-1]
    if n < 1:
        raise ValueError("Выборка не должна быть пустой.")
    z = stats.norm.ppf(1 - alpha / 2)
    mean, median = sample.mean(axis=-1), np.median(sample, axis=-1)
    return {
        "mean": (mean - z * scale * np.sqrt(2 / n), mean + z * scale * np.sqrt(2 / n)),
        "median": (median - z * scale / np.sqrt(n), median + z * scale / np.sqrt(n)),
    }


def laplace_median_coverage(n, alpha=0.05):
    validate_alpha(alpha)
    if n < 1 or n % 2 != 1:
        raise ValueError("Формула порядковой статистики требует нечётного n.")
    z = stats.norm.ppf(1 - alpha / 2)
    p = 0.5 * np.exp(-z / np.sqrt(n))
    return 1 - 2 * stats.binom.sf(n // 2, n, p)


def laplace_mean_coverage(n, alpha=0.05):
    validate_alpha(alpha)
    if n < 1:
        raise ValueError("Нужно n >= 1.")
    z = stats.norm.ppf(1 - alpha / 2)
    root = np.sqrt(n)
    bounds = (stats.gamma.ppf([1e-12, 1 - 1e-12], a=n) - n) / root

    def integrand(t):
        g = n + root * t
        if g <= 0:
            return 0.0
        prob = 2 * stats.norm.cdf(z * np.sqrt(n / g)) - 1
        return prob * np.exp(stats.gamma.logpdf(g, a=n)) * root

    return integrate.quad(integrand, *bounds, epsabs=2e-10, epsrel=2e-10)[0]
