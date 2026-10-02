"""
Sheaf H^0 fitters.

Given paired activations (a_i, b_i) drawn from paraphrase pairs, the kernel
of the sheaf Laplacian on the matching graph identifies content-stable
directions. These helpers cover the three restriction-map regimes used in
the paper: joint-PCA P, random orthonormal P, and the identity map (no
projection). A fourth helper takes an arbitrary edge set so that ring or
clique constructions can use the same machinery.
"""

from typing import Dict, Sequence, Tuple

import numpy as np
import torch


def _pca_basis(X: np.ndarray, edge_dim: int) -> np.ndarray:
    """(d, edge_dim) top principal directions of the rows of ``X``."""
    Xc = X - X.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Vt[:min(edge_dim, Vt.shape[0])].T


def _laplacian_eig_from_diffs(diffs: np.ndarray, P: np.ndarray):
    """Eigendecomposition (ascending) of (P^T D^T D P) / n for edge differences D."""
    Delta_k = diffs @ P
    return np.linalg.eigh((Delta_k.T @ Delta_k) / max(len(diffs), 1))


def _pull_back(P: np.ndarray, eigvecs_edge: np.ndarray, k: int, top: bool = False) -> np.ndarray:
    """Bottom-k (or top-k) edge-space eigenvectors mapped to native space and orthonormalised."""
    sel = eigvecs_edge[:, -k:] if top else eigvecs_edge[:, :k]
    basis, _ = np.linalg.qr(P @ sel)
    return basis


def edge_laplacian_eig(pairs_a: torch.Tensor, pairs_b: torch.Tensor, P: np.ndarray):
    """Eigendecomposition (ascending) of the edge-space sheaf Laplacian P^T C_W P."""
    return _laplacian_eig_from_diffs((pairs_a - pairs_b).numpy().astype(np.float32), P)


def joint_pca_projection(pairs_a: torch.Tensor, pairs_b: torch.Tensor, edge_dim: int = 128) -> np.ndarray:
    """(d, edge_dim) joint-PCA restriction map fit on the stacked pair activations."""
    return _pca_basis(torch.cat([pairs_a, pairs_b], dim=0).numpy().astype(np.float32), edge_dim)


def fit_h0_with_projection(
    pairs_a: torch.Tensor,
    pairs_b: torch.Tensor,
    P: np.ndarray,
    k: int = 20,
    top: bool = False,
) -> np.ndarray:
    """
    Bottom-k (or, with ``top=True``, top-k = H^1) eigenvectors of P^T C_W P
    pulled back to native space via P and orthonormalised.

    Args:
        pairs_a, pairs_b: (n, d) paired activations.
        P: (d, edge_dim) restriction map (orthonormal columns assumed).
        k: number of directions to return.

    Returns:
        (d, k) orthonormal basis in native space.
    """
    _, eigvecs_edge = edge_laplacian_eig(pairs_a, pairs_b, P)
    return _pull_back(P, eigvecs_edge, k, top)


def fit_h0_pca(
    pairs_a: torch.Tensor,
    pairs_b: torch.Tensor,
    k: int = 20,
    edge_dim: int = 128,
    top: bool = False,
) -> np.ndarray:
    """
    Sheaf H^0 with joint-PCA restriction map (``top=True`` gives H^1 instead).

    Fits P from the joint covariance of (a, b), then returns the bottom-k
    (top-k) eigenvectors of P^T C_W P pulled back to native space.
    """
    P = joint_pca_projection(pairs_a, pairs_b, edge_dim)
    return fit_h0_with_projection(pairs_a, pairs_b, P, k=k, top=top)


def fit_h0_random(
    pairs_a: torch.Tensor,
    pairs_b: torch.Tensor,
    k: int = 20,
    edge_dim: int = 128,
    seed: int = 0,
) -> np.ndarray:
    """
    Sheaf H^0 with a random orthonormal restriction map.

    Used to ablate the joint-PCA choice of P. The bottom-k eigenvectors of
    P^T C_W P with random P bound how much of the retrieval signal is owed
    to PCA versus the matching constraint itself.
    """
    d = pairs_a.shape[1]
    rng = np.random.default_rng(seed)
    P = rng.standard_normal((d, edge_dim)).astype(np.float32)
    P, _ = np.linalg.qr(P)
    return fit_h0_with_projection(pairs_a, pairs_b, P[:, :edge_dim], k=k)


def fit_h0_identity(
    pairs_a: torch.Tensor,
    pairs_b: torch.Tensor,
    k: int = 20,
) -> np.ndarray:
    """
    No projection: bottom-k eigenvectors of C_W in the full d-dim space.

    Uses SVD of the difference matrix so a (d, d) covariance never has to be
    formed. Columns of the returned basis include any null directions (zero
    within-pair variance) before strictly positive eigenvalues.
    """
    diffs = (pairs_a - pairs_b).numpy().astype(np.float32)
    n = diffs.shape[0]
    U, S, _ = np.linalg.svd(diffs.T, full_matrices=True)
    eig = np.zeros(U.shape[0], dtype=np.float32)
    eig[:len(S)] = S ** 2 / max(n, 1)
    order = np.argsort(eig)
    return U[:, order[:k]].astype(np.float32)


def fit_h0_from_edges(
    activations: Dict[str, np.ndarray],
    edges: Sequence[Tuple[str, str]],
    k: int = 20,
    edge_dim: int = 128,
) -> np.ndarray:
    """
    Sheaf H^0 with joint-PCA P on an arbitrary graph (multi-paraphrase rings,
    cliques, etc.). Activations are addressed by node name; edges list pairs
    of node names. Joint PCA is fit on the union of node activations.
    """
    names = list(activations.keys())
    name_to_idx = {n: i for i, n in enumerate(names)}
    X = np.stack([activations[n] for n in names]).astype(np.float32)
    P = _pca_basis(X, edge_dim)
    diffs = np.array([
        X[name_to_idx[a]] - X[name_to_idx[b]]
        for a, b in edges if a in name_to_idx and b in name_to_idx
    ], dtype=np.float32)
    _, eigvecs_edge = _laplacian_eig_from_diffs(diffs, P)
    return _pull_back(P, eigvecs_edge, k)
