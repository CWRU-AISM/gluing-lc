"""
Sheaf-Laplacian and Fisher decompositions.

Operates on paraphrase activation differences in the projected edge
space, returning the H^0 / H^1 eigenbases consumed by the steering
methods in :mod:`sheafint.steering.methods`.
"""

from typing import Dict
import numpy as np


def _split_spectrum(
    eigvals: np.ndarray,
    eigvecs_edge: np.ndarray,
    projection: np.ndarray,
    background: np.ndarray,
    k: int,
) -> Dict:
    """Sort the edge-space spectrum and derive H^0 / H^1 bases and coordinate rankings.

    ``h0_vecs`` / ``h1_vecs`` are the bottom-k / top-k eigenvectors pulled back
    through ``projection`` with unit-norm columns. ``h1_dims`` ranks native
    coordinates by their eigenvalue-weighted loading on the top-k
    eigenvectors; ``var_h1`` ranks them by background variance.
    """
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

    h1_dim_score = np.sum(
        (projection @ eigvecs_edge[:, -k:]) ** 2 * eigvals[-k:][np.newaxis, :],
        axis=1,
    )
    return {
        'h0_vecs': h0_vecs,
        'h1_vecs': h1_vecs,
        'h1_dims': np.argsort(h1_dim_score)[-k:][::-1].tolist(),
        'var_h1': np.argsort(np.var(background, axis=0))[-k:][::-1].tolist(),
        'eigvals': eigvals,
    }


def sheaf_laplacian_decomposition(
    s_left: np.ndarray,
    s_right: np.ndarray,
    background: np.ndarray,
    projection: np.ndarray,
    k: int = 20,
) -> Dict:
    """Eigendecomposition of L = (1/N) sum_i (P delta_i)(P delta_i)^T.

    Small eigenvalues span the H^0 (consistent) subspace; large ones the
    H^1-like variation. ``h0_dim`` counts eigenvalues below 1% of the largest.
    """
    differences = s_left @ projection - s_right @ projection
    laplacian = (differences.T @ differences) / s_left.shape[0]
    eigvals, eigvecs_edge = np.linalg.eigh(laplacian)

    out = _split_spectrum(eigvals, eigvecs_edge, projection, background, k)
    eigvals = out['eigvals']
    max_eig = eigvals[-1] if len(eigvals) else 1.0
    h0_dim = int(np.sum(eigvals < 0.01 * max_eig))
    out['h0_dim'] = h0_dim
    out['spectral_gap'] = float(eigvals[-1] / (eigvals[h0_dim] + 1e-10)) if h0_dim < len(eigvals) else 0.0
    out['method'] = 'sheaf_laplacian'
    return out


def fisher_decomposition(
    s_left: np.ndarray,
    s_right: np.ndarray,
    background: np.ndarray,
    projection: np.ndarray,
    k: int = 20,
) -> Dict:
    """Generalized eigenproblem L_within v = lambda C_total v in the edge space."""
    edge_dim = projection.shape[1]

    differences = s_left @ projection - s_right @ projection
    diff_centered = differences - differences.mean(axis=0)
    within = (diff_centered.T @ diff_centered) / s_left.shape[0]

    bg_edge = background @ projection
    bg_centered = bg_edge - bg_edge.mean(axis=0)
    total = (bg_centered.T @ bg_centered) / background.shape[0]
    total_reg = total + 1e-4 * np.trace(total) / edge_dim * np.eye(edge_dim)

    try:
        L_inv = np.linalg.inv(np.linalg.cholesky(total_reg))
        eigvals, eigvecs_M = np.linalg.eigh(L_inv @ within @ L_inv.T)
        eigvecs_edge = L_inv.T @ eigvecs_M
    except np.linalg.LinAlgError:
        eigvals, eigvecs_edge = np.linalg.eigh(within)

    out = _split_spectrum(eigvals, eigvecs_edge, projection, background, k)
    out['method'] = 'fisher_lda'
    return out
