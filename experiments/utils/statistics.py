# Statistical helpers (bootstrap CIs, McNemar's test).

from typing import Iterable, Tuple
import numpy as np
from scipy.stats import binomtest, chi2 as _chi2_dist


def bootstrap_ci(
    values: Iterable[float],
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    # Bootstrap (mean, lo, hi) for a 1-D array of samples.
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
    # McNemar's test for paired binary outcomes; uses exact binomial when n_disc < 25.
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
