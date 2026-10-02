"""
Statistical helpers shared across experiments.

Provides a bootstrap mean confidence interval, McNemar's test for paired
binary outcomes (used to compare steering accuracies), and an exact
permutation Spearman test for small model-level correlations.
"""

import itertools
import math
from typing import Iterable, Sequence, Tuple

import numpy as np
from scipy.stats import binomtest, chi2 as _chi2_dist, rankdata


def bootstrap_ci(
    values: Iterable[float],
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Bootstrap mean and percentile CI for a 1-D array of samples.

    Returns ``(mean, lo, hi)`` for the requested confidence level. An empty
    input collapses to ``(0, 0, 0)`` so callers do not need to guard against
    silent NaN propagation.
    """
    rng = np.random.default_rng(seed)
    arr = np.asarray(list(values), dtype=float)
    n = len(arr)
    if n == 0:
        return 0.0, 0.0, 0.0
    boot_means = np.array([
        arr[rng.integers(0, n, size=n)].mean()
        for _ in range(n_boot)
    ])
    alpha = (1 - ci) / 2
    lo = float(np.percentile(boot_means, alpha * 100))
    hi = float(np.percentile(boot_means, (1 - alpha) * 100))
    return float(arr.mean()), lo, hi


def mcnemar_test(s1: Iterable[int], s2: Iterable[int]) -> float:
    """
    McNemar's test for paired binary outcomes.

    Falls back to the exact binomial distribution when fewer than 25
    discordant pairs are observed; otherwise uses the continuity-corrected
    chi-square approximation.
    """
    a = np.asarray(list(s1))
    b = np.asarray(list(s2))
    only_a = int(((a == 1) & (b == 0)).sum())
    only_b = int(((a == 0) & (b == 1)).sum())
    n_disc = only_a + only_b
    if n_disc == 0:
        return 1.0
    if n_disc < 25:
        return float(binomtest(only_a, n_disc, 0.5).pvalue)
    chi2 = (abs(only_a - only_b) - 1) ** 2 / n_disc
    return float(1 - _chi2_dist.cdf(chi2, df=1))


def spearman_exact(x: Sequence[float], y: Sequence[float]) -> Tuple[float, float]:
    """
    Spearman rho between ``x`` and ``y`` with the exact one-sided permutation
    p-value P(rho_perm >= rho_obs) over all n! rankings of ``y``.

    Used for the harmonic-mass vs steering-fragility correlation (n <= 9, so
    the enumeration is at most 362,880 rankings).
    """
    rx, ry = rankdata(x), rankdata(y)
    corr = lambda a, b: float(np.corrcoef(a, b)[0, 1])
    obs = corr(rx, ry)
    hits = sum(corr(rx, np.asarray(perm)) >= obs - 1e-12 for perm in itertools.permutations(ry))
    return obs, hits / math.factorial(len(ry))
