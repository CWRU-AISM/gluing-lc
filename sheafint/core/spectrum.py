# Sheaf Laplacian spectrum and H0 projection diagnostics.

from typing import Dict, Optional
import warnings
import numpy as np
import torch
from .svd import randomized_truncated_svd


def compute_laplacian_spectrum(delta0: torch.Tensor, tol: float, device: str, k: int = 50) -> Dict:
    # Eigendecomposition of the node Laplacian L = delta0^T delta0.
    L_node = delta0.T @ delta0

    eigenvectors: Optional[torch.Tensor]
    try:
        eigenvalues_t, eigenvectors_t = torch.linalg.eigh(L_node)
        eigenvalues = eigenvalues_t.cpu().numpy()
        eigenvectors = eigenvectors_t.cpu()
    except Exception as exc:
        warnings.warn(f"Full eigendecomposition failed: {exc}. Using randomized SVD.")
        _, S, _ = randomized_truncated_svd(delta0, min(k, min(delta0.shape) - 1), device=device)
        eigenvalues = (S.cpu().numpy() ** 2)
        eigenvectors = None

    sorted_idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[sorted_idx]
    if eigenvectors is not None:
        eigenvectors = eigenvectors[:, sorted_idx]

    h0_dim = int(np.sum(eigenvalues < tol))

    nonzero = eigenvalues[eigenvalues >= tol]
    spectral_gap = float(nonzero[0]) if len(nonzero) > 0 else 0.0

    positive = eigenvalues[eigenvalues > tol]
    if len(positive) > 0:
        probs = positive / positive.sum()
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        effective_rank = float(np.exp(entropy))
    else:
        effective_rank = 0.0

    condition_number = float(nonzero[-1] / nonzero[0]) if len(nonzero) > 1 else 1.0

    return {
        'eigenvalues': eigenvalues.tolist(),
        'eigenvectors': eigenvectors,
        'h0_dim': h0_dim,
        'spectral_gap': spectral_gap,
        'effective_rank': effective_rank,
        'condition_number': condition_number,
        'trace': float(np.sum(eigenvalues)),
        'top_10_eigenvalues': eigenvalues[-10:].tolist() if len(eigenvalues) >= 10 else eigenvalues.tolist(),
    }


def project_signal_onto_h0(
    signal: torch.Tensor,
    h0_basis: Optional[torch.Tensor],
    delta0: torch.Tensor,
    edge_dim: int,
    nodes,
    node_to_idx,
    tol: float,
) -> Dict:
    # Project a C^0 signal onto the kernel of delta0 (the H0 subspace).
    if h0_basis is None or h0_basis.shape[1] == 0:
        s_h0 = torch.zeros_like(signal)
    else:
        H0 = h0_basis.to(signal.device)
        s_h0 = H0 @ (H0.T @ signal)

    total = torch.norm(signal).item() ** 2 + 1e-10
    h0_energy = torch.norm(s_h0).item() ** 2

    perp = signal - s_h0
    perp_energy = torch.norm(perp).item() ** 2

    per_node_fractions = {}
    for node in nodes:
        idx = node_to_idx[node]
        start = idx * edge_dim
        end = start + edge_dim
        node_total = torch.norm(signal[start:end]).item() ** 2 + 1e-10
        node_h0 = torch.norm(s_h0[start:end]).item() ** 2
        per_node_fractions[node] = node_h0 / node_total

    coboundary_residual = torch.norm(delta0 @ s_h0).item()

    return {
        'h0_fraction': h0_energy / total,
        'perp_fraction': perp_energy / total,
        'h0_energy': h0_energy,
        'total_energy': total,
        'per_node_fractions': per_node_fractions,
        'coboundary_of_h0_projection': coboundary_residual,
        'is_valid': coboundary_residual < tol * 10,
    }
