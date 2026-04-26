# Restriction-map estimators used by steering experiments (PCA and Fisher / contrastive).

from typing import Tuple
import numpy as np


def joint_pca(s_left: np.ndarray, s_right: np.ndarray, edge_dim: int) -> Tuple[np.ndarray, float]:
    # Joint PCA over both halves of paraphrase pairs. Returns projection P and explained variance.
    stacked = np.vstack([s_left, s_right])
    centered = stacked - stacked.mean(axis=0)
    rank = min(edge_dim, centered.shape[0] - 1, s_left.shape[1])
    _, sigma, Vt = np.linalg.svd(centered, full_matrices=False)
    P = Vt[:rank].T
    explained = (sigma[:rank] ** 2).sum() / (sigma ** 2).sum()
    return P, float(explained)


def contrastive_fisher(
    s_left: np.ndarray,
    s_right: np.ndarray,
    background: np.ndarray,
    edge_dim: int,
    base_projection: np.ndarray = None,
) -> Tuple[np.ndarray, float, np.ndarray]:
    # Fisher discriminant in the joint-PCA subspace, providing a contrastive restriction map.
    n, _ = s_left.shape

    if base_projection is None:
        base_projection, _ = joint_pca(s_left, s_right, edge_dim)
    k_pca = base_projection.shape[1]

    s_left_pca = s_left @ base_projection
    s_right_pca = s_right @ base_projection
    background_pca = background @ base_projection

    diffs = s_left_pca - s_right_pca
    diffs_centered = diffs - diffs.mean(axis=0)
    within = (diffs_centered.T @ diffs_centered) / n

    bg_centered = background_pca - background_pca.mean(axis=0)
    total = (bg_centered.T @ bg_centered) / background.shape[0]
    between = total - within

    eps = 1e-4 * np.trace(within) / k_pca
    within_reg = within + eps * np.eye(k_pca)

    try:
        L = np.linalg.cholesky(within_reg)
        L_inv = np.linalg.inv(L)
        M = L_inv @ between @ L_inv.T
        eigvals, eigvecs_M = np.linalg.eigh(M)
        eigvecs_pca = L_inv.T @ eigvecs_M
    except np.linalg.LinAlgError:
        eigvals, eigvecs_pca = np.linalg.eigh(between)

    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs_pca = eigvecs_pca[:, order]

    P = base_projection @ eigvecs_pca
    explained = float(np.sum(np.abs(eigvals[:k_pca])) / (np.sum(np.abs(eigvals)) + 1e-10))
    return P, explained, eigvals
