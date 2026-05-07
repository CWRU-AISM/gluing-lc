"""
Sheaf-Laplacian and Fisher decompositions.

Operates on paraphrase activation differences in the projected edge
space, returning the H^0 / H^1 eigenbases consumed by the steering
methods in :mod:`sheafint.steering.methods`.
"""

from typing import Dict
import numpy as np


def sheaf_laplacian_decomposition(
    s_left: np.ndarray,
    s_right: np.ndarray,
    background: np.ndarray,
    projection: np.ndarray,
    k: int = 20,
) -> Dict:
    # Eigendecomposition of L = (1/N) sum_i (P delta_i)(P delta_i)^T.
    # Small eigenvalues correspond to the H0 (consistent) subspace; large ones to H1-like variation.
    n_pairs, _ = s_left.shape
    edge_dim = projection.shape[1]

    s_left_edge = s_left @ projection
    s_right_edge = s_right @ projection

    differences = s_left_edge - s_right_edge
    laplacian = (differences.T @ differences) / n_pairs

    eigvals, eigvecs_edge = np.linalg.eigh(laplacian)
    order = np.argsort(eigvals)
    eigvals = eigvals[order]
    eigvecs_edge = eigvecs_edge[:, order]

    h0_vecs = projection @ eigvecs_edge[:, :k]
    h1_vecs = projection @ eigvecs_edge[:, -k:]

    for vecs in (h0_vecs, h1_vecs):
        for col in range(vecs.shape[1]):
            norm = np.linalg.norm(vecs[:, col])
            if norm > 0:
                vecs[:, col] /= norm

    top_k_vals = eigvals[-k:]
    h1_dim_score = np.sum(
        (projection @ eigvecs_edge[:, -k:]) ** 2 * top_k_vals[np.newaxis, :],
        axis=1,
    )
    h1_dims = np.argsort(h1_dim_score)[-k:][::-1].tolist()

    var_h1 = np.argsort(np.var(background, axis=0))[-k:][::-1].tolist()

    max_eig = eigvals[-1] if len(eigvals) else 1.0
    rel_threshold = 0.01 * max_eig
    h0_dim = int(np.sum(eigvals < rel_threshold))
    spectral_gap = float(eigvals[-1] / (eigvals[h0_dim] + 1e-10)) if h0_dim < len(eigvals) else 0.0

    return {
        'h0_vecs': h0_vecs,
        'h1_vecs': h1_vecs,
        'h1_dims': h1_dims,
        'var_h1': var_h1,
        'eigvals': eigvals,
        'h0_dim': h0_dim,
        'spectral_gap': spectral_gap,
        'method': 'sheaf_laplacian',
    }


def fisher_decomposition(
    s_left: np.ndarray,
    s_right: np.ndarray,
    background: np.ndarray,
    projection: np.ndarray,
    k: int = 20,
) -> Dict:
    # Generalized eigenvalue problem L_within v = lambda C_total v.
    n_pairs, _ = s_left.shape
    edge_dim = projection.shape[1]

    s_left_edge = s_left @ projection
    s_right_edge = s_right @ projection
    bg_edge = background @ projection

    differences = s_left_edge - s_right_edge
    diff_centered = differences - differences.mean(axis=0)
    within = (diff_centered.T @ diff_centered) / n_pairs

    bg_centered = bg_edge - bg_edge.mean(axis=0)
    total = (bg_centered.T @ bg_centered) / background.shape[0]

    eps = 1e-4 * np.trace(total) / edge_dim
    total_reg = total + eps * np.eye(edge_dim)

    try:
        L_chol = np.linalg.cholesky(total_reg)
        L_inv = np.linalg.inv(L_chol)
        M = L_inv @ within @ L_inv.T
        eigvals, eigvecs_M = np.linalg.eigh(M)
        eigvecs_edge = L_inv.T @ eigvecs_M
    except np.linalg.LinAlgError:
        eigvals, eigvecs_edge = np.linalg.eigh(within)

    order = np.argsort(eigvals)
    eigvals = eigvals[order]
    eigvecs_edge = eigvecs_edge[:, order]

    h0_vecs = projection @ eigvecs_edge[:, :k]
    h1_vecs = projection @ eigvecs_edge[:, -k:]

    for vecs in (h0_vecs, h1_vecs):
        for col in range(vecs.shape[1]):
            norm = np.linalg.norm(vecs[:, col])
            if norm > 0:
                vecs[:, col] /= norm

    top_k_vals = eigvals[-k:]
    h1_dim_score = np.sum(
        (projection @ eigvecs_edge[:, -k:]) ** 2 * top_k_vals[np.newaxis, :],
        axis=1,
    )
    h1_dims = np.argsort(h1_dim_score)[-k:][::-1].tolist()
    var_h1 = np.argsort(np.var(background, axis=0))[-k:][::-1].tolist()

    return {
        'h0_vecs': h0_vecs,
        'h1_vecs': h1_vecs,
        'h1_dims': h1_dims,
        'var_h1': var_h1,
        'eigvals': eigvals,
        'method': 'fisher_lda',
    }
