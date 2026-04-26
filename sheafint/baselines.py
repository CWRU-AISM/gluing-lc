# Baseline similarity / consistency methods for comparison with sheaf metrics.

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import torch


@dataclass
class BaselineMetrics:
    # Container for baseline comparison metrics.
    method: str
    consistency_score: float
    random_baseline: float
    improvement_pct: float
    raw_scores: Optional[torch.Tensor] = None


def cosine_similarity_consistency(features_a: torch.Tensor, features_b: torch.Tensor) -> float:
    # Mean cosine similarity between paired feature vectors.
    a_norm = features_a / (features_a.norm(dim=1, keepdim=True) + 1e-8)
    b_norm = features_b / (features_b.norm(dim=1, keepdim=True) + 1e-8)
    return (a_norm * b_norm).sum(dim=1).mean().item()


def cosine_consistency_energy(features_a: torch.Tensor, features_b: torch.Tensor) -> float:
    # Energy form of cosine consistency (lower = more consistent).
    return 1.0 - cosine_similarity_consistency(features_a, features_b)


def linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    # Linear CKA (Kornblith et al., 2019) using the kernel-trick form.
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)
    n = X.shape[0]
    XtY = X.T @ Y
    hsic_xy = (XtY ** 2).sum() / ((n - 1) ** 2)
    XtX = X.T @ X
    hsic_xx = (XtX ** 2).sum() / ((n - 1) ** 2)
    YtY = Y.T @ Y
    hsic_yy = (YtY ** 2).sum() / ((n - 1) ** 2)
    return (hsic_xy / (torch.sqrt(hsic_xx * hsic_yy) + 1e-8)).item()


def cka_consistency(features_a: torch.Tensor, features_b: torch.Tensor) -> float:
    return linear_cka(features_a, features_b)


def cka_consistency_energy(features_a: torch.Tensor, features_b: torch.Tensor) -> float:
    return 1.0 - cka_consistency(features_a, features_b)


def mse_consistency_energy(features_a: torch.Tensor, features_b: torch.Tensor) -> float:
    # Mean squared difference between paired features.
    return ((features_a - features_b) ** 2).mean().item()


_METHOD_FNS = {
    'cosine': cosine_consistency_energy,
    'cka': cka_consistency_energy,
    'mse': mse_consistency_energy,
}


def run_baseline_comparison(
    features: Dict[str, torch.Tensor],
    pair_overlaps: Dict[Tuple[str, str], torch.Tensor],
    methods: List[str] = ('cosine', 'cka', 'mse'),
) -> Dict[str, BaselineMetrics]:
    # Compute consistency energy under each method against a shuffled-pair random baseline.
    results: Dict[str, BaselineMetrics] = {}

    for (node_i, node_j), indices in pair_overlaps.items():
        if node_i not in features or node_j not in features:
            continue

        Fi = features[node_i][indices]
        Fj = features[node_j][indices]
        Fi_rand = Fi[torch.randperm(Fi.shape[0])]

        for method in methods:
            if method not in _METHOD_FNS:
                raise ValueError(f"Unknown method: {method}")
            fn = _METHOD_FNS[method]
            actual = fn(Fi, Fj)
            random_score = fn(Fi_rand, Fj)
            improvement = (random_score - actual) / (random_score + 1e-8) * 100
            results[method] = BaselineMetrics(
                method=method,
                consistency_score=actual,
                random_baseline=random_score,
                improvement_pct=improvement,
            )

    return results
