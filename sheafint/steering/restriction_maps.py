"""
Restriction-map estimators for steering experiments.

``joint_pca`` is the base map. ``cca`` and ``contrastive`` refine it inside the
joint-PCA space (canonical correlation between pair members, and a Fisher
discriminant of between- vs within-pair variation) and return the composed
d x k map whose columns order the edge space for the sheaf or Fisher
decomposition.
"""

import warnings
from typing import Tuple

import numpy as np


def joint_pca(s_left: np.ndarray, s_right: np.ndarray, edge_dim: int) -> Tuple[np.ndarray, float]:
    """Joint PCA over both halves of paraphrase pairs; returns (P, explained variance ratio)."""
    stacked = np.vstack([s_left, s_right])
    centered = stacked - stacked.mean(axis=0)
    rank = min(edge_dim, centered.shape[0] - 1, s_left.shape[1])
    _, sigma, Vt = np.linalg.svd(centered, full_matrices=False)
    P = Vt[:rank].T
    explained = (sigma[:rank] ** 2).sum() / (sigma ** 2).sum()
    return P, float(explained)


def cca(s_left: np.ndarray, s_right: np.ndarray, P_pca: np.ndarray, eps: float = 1e-4) -> Tuple[np.ndarray, np.ndarray]:
    """CCA between pair members in joint-PCA space, composed with ``P_pca``.

    Returns the d x k map with unit-norm columns ordered from least to most
    correlated (low correlation = H^1-like), and the canonical correlations.
    Falls back to ``P_pca`` if the covariance whitening fails.
    """
    n = s_left.shape[0]
    k = P_pca.shape[1]
    a = s_left @ P_pca
    b = s_right @ P_pca
    a = a - a.mean(axis=0)
    b = b - b.mean(axis=0)
    C11 = (a.T @ a) / n + eps * np.eye(k)
    C22 = (b.T @ b) / n + eps * np.eye(k)
    C12 = (a.T @ b) / n
    try:
        w11, V11 = np.linalg.eigh(C11)
        C11_inv_sqrt = V11 @ np.diag(1.0 / np.sqrt(np.maximum(w11, 1e-10))) @ V11.T
        w22, V22 = np.linalg.eigh(C22)
        C22_inv = V22 @ np.diag(1.0 / np.maximum(w22, 1e-10)) @ V22.T
        corr_sq, vecs = np.linalg.eigh(C11_inv_sqrt @ C12 @ C22_inv @ C12.T @ C11_inv_sqrt)
    except np.linalg.LinAlgError as exc:
        warnings.warn(f'CCA failed ({exc}); using the joint-PCA map')
        return P_pca, np.ones(k)

    order = np.argsort(corr_sq)
    P = P_pca @ (C11_inv_sqrt @ vecs[:, order])
    for i in range(P.shape[1]):
        norm = np.linalg.norm(P[:, i])
        if norm > 1e-10:
            P[:, i] /= norm
    return P, np.sqrt(np.clip(corr_sq[order], 0, 1))


def contrastive(
    s_left: np.ndarray,
    s_right: np.ndarray,
    background: np.ndarray,
    P_pca: np.ndarray,
) -> Tuple[np.ndarray, float, np.ndarray]:
    """Fisher discriminant of between- vs within-pair covariance in joint-PCA space.

    Returns the composed d x k map (columns by decreasing discriminant
    eigenvalue), the explained-variance ratio, and the sorted eigenvalues.
    """
    n = s_left.shape[0]
    k = P_pca.shape[1]
    diffs = s_left @ P_pca - s_right @ P_pca
    diffs = diffs - diffs.mean(axis=0)
    C_within = (diffs.T @ diffs) / n
    bg = background @ P_pca
    bg = bg - bg.mean(axis=0)
    C_between = (bg.T @ bg) / background.shape[0] - C_within
    C_within_reg = C_within + 1e-4 * np.trace(C_within) / k * np.eye(k)
    try:
        L_inv = np.linalg.inv(np.linalg.cholesky(C_within_reg))
        eigvals, vecs_M = np.linalg.eigh(L_inv @ C_between @ L_inv.T)
        vecs = L_inv.T @ vecs_M
    except np.linalg.LinAlgError:
        warnings.warn('Cholesky of the within-pair covariance failed; using a plain eigendecomposition')
        eigvals, vecs = np.linalg.eigh(C_between)

    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    explained = np.sum(np.abs(eigvals[:k])) / (np.sum(np.abs(eigvals)) + 1e-10)
    return P_pca @ vecs[:, order], float(explained), eigvals
