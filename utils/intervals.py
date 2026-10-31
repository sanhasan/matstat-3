"""Точные F-интервалы и асимптотические интервалы для модели Лапласа."""

import numpy as np
from scipy import integrate, stats


def validate_alpha(alpha):
    if not 0 < alpha < 1:
        raise ValueError("Требуется 0 < alpha < 1.")


def variance_ratio_interval(var_x, var_y, n, m, alpha=0.05):
    """Входные дисперсии несмещённые (ddof=1), последние оси — повторения."""
    validate_alpha(alpha)
    if n < 2 or m < 2:
        raise ValueError("Для оценки дисперсии нужны n, m >= 2.")
    var_x, var_y = np.asarray(var_x), np.asarray(var_y)
    if np.any(var_x < 0) or np.any(var_y <= 0):
        raise ValueError("Нужны var_x >= 0 и var_y > 0.")
    ratio = var_x / var_y
    q_low, q_high = stats.f.ppf([alpha / 2, 1 - alpha / 2], n - 1, m - 1)
    return ratio / q_high, ratio / q_low


def ratio_from_samples(x, y, alpha=0.05):
    """Каждая строка x и y — отдельная независимая выборка."""
    x, y = np.asarray(x), np.asarray(y)
    return variance_ratio_interval(
        x.var(axis=-1, ddof=1), y.var(axis=-1, ddof=1),
        x.shape[-1], y.shape[-1], alpha,
    )


def ratio_expected_bounds(n, m, tau=2.0, alpha=0.05):
    """E[L], E[U], E[U-L]; при m <= 3 ожидания бесконечны."""
    validate_alpha(alpha)
    if n < 2 or m < 2 or tau <= 0:
        raise ValueError("Нужны n, m >= 2 и tau > 0.")
    if m <= 3:
        return np.inf, np.inf, np.inf
    q_low, q_high = stats.f.ppf([alpha / 2, 1 - alpha / 2], n - 1, m - 1)
    mean_ratio = tau * (m - 1) / (m - 3)
    low, high = mean_ratio / q_high, mean_ratio / q_low
    return low, high, high - low


def laplace_intervals(sample, alpha=0.05, scale=1.0):
    """Известный масштаб b: mean ± z*b*sqrt(2/n), median ± z*b/sqrt(n)."""
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
    """Точное покрытие асимптотического интервала при нечётном n, не симуляция."""
    validate_alpha(alpha)
    if n < 1 or n % 2 != 1:
        raise ValueError("Формула порядковой статистики требует нечётного n.")
    z = stats.norm.ppf(1 - alpha / 2)
    p = 0.5 * np.exp(-z / np.sqrt(n))
    return 1 - 2 * stats.binom.sf(n // 2, n, p)


def laplace_median_exact_halfwidth(n, alpha=0.05, scale=1.0):
    """Дополнительный точный интервал median ± h через Beta(k, k), нечётное n."""
    validate_alpha(alpha)
    if n < 1 or n % 2 != 1 or scale <= 0:
        raise ValueError("Нужны положительный масштаб и нечётное n >= 1.")
    k = (n+1)//2
    p = stats.beta.ppf(alpha/2, k, k)
    return -scale * np.log(2*p)


def laplace_mean_coverage(n, alpha=0.05):
    """Точное конечновыборочное покрытие через смесь нормального и Gamma.

    S_n | G ~ N(0, 2G), G ~ Gamma(n, 1). Интегрирование после
    стандартизации G; суммарная отброшенная масса не больше 2e-12.
    """
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
