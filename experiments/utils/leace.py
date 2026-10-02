"""
LEACE comparisons: eraser fits and the 60-fact centroid lookup.

``fit_leace_eraser`` and ``top_pcs`` serve the held-out CounterFact
comparison; ``centroid_lookup``, ``paraphrase_pairs`` and
``random_first_paraphrases`` the templated 60-fact table.
"""

import numpy as np
import torch
from concept_erasure import LeaceFitter


def fit_leace_eraser(H_train_a: torch.Tensor, seed: int):
    """LEACE eraser fit on the train activations (z = 1) against a seeded permutation of them (z = 0)."""
    n = H_train_a.shape[0]
    neg_perm = np.random.default_rng(seed).permutation(n)
    X = torch.cat([H_train_a, H_train_a[neg_perm]], dim=0)
    z = torch.cat([torch.ones(n), torch.zeros(n)]).long()
    return LeaceFitter.fit(X, z).eraser


def fit_pair_eraser(positives: torch.Tensor, negatives: torch.Tensor):
    """LEACE eraser for paraphrase-pair members (z = 1) vs non-paraphrase prompts (z = 0)."""
    n = len(positives)
    X = torch.cat([positives, negatives[:n]], dim=0)
    z = torch.cat([torch.ones(n), torch.zeros(n)]).long()
    return LeaceFitter.fit(X, z).eraser


def top_pcs(X: np.ndarray, k: int) -> np.ndarray:
    """Orthonormal (d, k) top principal directions of the rows of ``X``."""
    Xc = X - X.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    basis, _ = np.linalg.qr(Vt[:k].T)
    return basis


def centroid_lookup(features: np.ndarray, fact_labels: np.ndarray, basis=None) -> float:
    """Top-1 accuracy of each row against cosine-nearest per-fact centroids (rows included)."""
    proj = features @ basis if basis is not None else features
    facts = np.unique(fact_labels)
    centroids = np.stack([proj[fact_labels == f].mean(axis=0) for f in facts])
    centroids = centroids / (np.linalg.norm(centroids, axis=-1, keepdims=True) + 1e-8)
    proj = proj / (np.linalg.norm(proj, axis=-1, keepdims=True) + 1e-8)
    return float((facts[(proj @ centroids.T).argmax(axis=1)] == fact_labels).mean())


def paraphrase_pairs(H: np.ndarray, n_facts: int, n_par: int):
    """Consecutive paraphrase pairs (p0, p1), (p1, p2) of every fact, rows of ``H`` grouped by fact."""
    idx = [(i * n_par + j, i * n_par + j + 1) for i in range(n_facts) for j in range(min(2, n_par - 1))]
    a, b = zip(*idx)
    return torch.from_numpy(H[list(a)]), torch.from_numpy(H[list(b)])


def random_first_paraphrases(H: np.ndarray, n_facts: int, n_par: int, n: int) -> torch.Tensor:
    """First paraphrase of ``n`` random facts (the LEACE negative class)."""
    rng = np.random.default_rng(0)
    rows = []
    for _ in range(n):
        i = rng.integers(0, n_facts)
        rng.integers(0, n_facts)  # the paired draw keeps the published RNG stream
        rows.append(H[i * n_par])
    return torch.from_numpy(np.stack(rows))
