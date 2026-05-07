"""
Sheaf H^0 fitters.

Given paired activations (a_i, b_i) drawn from paraphrase pairs, the kernel
of the sheaf Laplacian on the matching graph identifies content-stable
directions. These helpers cover the three restriction-map regimes used in
the paper: joint-PCA P, random orthonormal P, and the identity map (no
projection). A fourth helper takes an arbitrary edge set so that ring or
clique constructions can use the same machinery.
"""

from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch


def fit_h0_with_projection(
    pairs_a: torch.Tensor,
    pairs_b: torch.Tensor,
    P: np.ndarray,
    k: int = 20,
) -> np.ndarray:
    """
    Bottom-k eigenvectors of P^T C_W P pulled back to native space via P.

    Args:
        pairs_a, pairs_b: (n, d) paired activations.
        P: (d, edge_dim) restriction map (orthonormal columns assumed).
        k: number of H^0 directions to return.

    Returns:
        (d, k) orthonormal basis for H^0 in native space.
    """
    n = pairs_a.shape[0]
    diffs = (pairs_a - pairs_b).numpy().astype(np.float32)
    Delta_k = diffs @ P
    L = (Delta_k.T @ Delta_k) / max(n, 1)
    _, eigvecs_edge = np.linalg.eigh(L)
    h0_edge = eigvecs_edge[:, :k]
    h0_native = P @ h0_edge
    h0_native, _ = np.linalg.qr(h0_native)
    return h0_native


def fit_h0_pca(
    pairs_a: torch.Tensor,
    pairs_b: torch.Tensor,
    k: int = 20,
    edge_dim: int = 128,
) -> np.ndarray:
    """
    Sheaf H^0 with joint-PCA restriction map.

    Fits P from the joint covariance of (a, b), then returns the bottom-k
    eigenvectors of P^T C_W P pulled back to native space.
    """
    X = torch.cat([pairs_a, pairs_b], dim=0).numpy().astype(np.float32)
    Xc = X - X.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    edge_dim = min(edge_dim, Vt.shape[0])
    P = Vt[:edge_dim].T
    return fit_h0_with_projection(pairs_a, pairs_b, P, k=k)


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
    Xc = X - X.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    edge_dim = min(edge_dim, Vt.shape[0])
    P = Vt[:edge_dim].T

    diffs: List[np.ndarray] = []
    for a, b in edges:
        if a in name_to_idx and b in name_to_idx:
            diffs.append(X[name_to_idx[a]] - X[name_to_idx[b]])
    diffs_arr = np.array(diffs, dtype=np.float32)

    Delta_k = diffs_arr @ P
    L = (Delta_k.T @ Delta_k) / max(len(diffs_arr), 1)
    _, eigvecs_edge = np.linalg.eigh(L)
    h0_edge = eigvecs_edge[:, :k]
    h0_native = P @ h0_edge
    h0_native, _ = np.linalg.qr(h0_native)
    return h0_native
