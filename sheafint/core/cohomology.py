"""
Cohomology dimension computation and Hodge decomposition.

Returns ``dim H^0 = dim ker delta_0`` and ``dim H^1 = dim ker delta_1
- dim im delta_0`` along with the exact / harmonic / coexact split of a
1-cochain.
"""

from typing import Dict, Optional
import torch
from .svd import numerical_rank


def compute_cohomology_dims(
    delta0: torch.Tensor,
    delta1: Optional[torch.Tensor],
    svd_rank: int,
    tol: float,
    use_randomized_svd: bool,
    device: str,
) -> Dict[str, int]:
    # Cohomology dimensions:
    #   H0 = dim(ker delta0)
    #   H1 = dim(ker delta1) - dim(im delta0)
    rank_d0 = numerical_rank(delta0, svd_rank, tol, use_randomized_svd, device=device)
    dim_C0 = delta0.shape[1]
    dim_C1 = delta0.shape[0]

    H0_dim = dim_C0 - rank_d0

    if delta1 is not None:
        rank_d1 = numerical_rank(delta1, svd_rank, tol, use_randomized_svd, device=device)
        H1_dim = max(0, (dim_C1 - rank_d1) - rank_d0)
    else:
        rank_d1 = 0
        H1_dim = dim_C1 - rank_d0

    return {
        'H0_dim': H0_dim,
        'H1_dim': H1_dim,
        'rank_delta0': rank_d0,
        'rank_delta1': rank_d1,
        'dim_C0': dim_C0,
        'dim_C1': dim_C1,
    }


def hodge_decomposition(
    edge_section: torch.Tensor,
    delta0: torch.Tensor,
    delta1: Optional[torch.Tensor],
) -> Dict[str, float]:
    # Decompose a 1-cochain into exact, harmonic, and coexact components and report energy fractions.
    d0_pinv = torch.linalg.pinv(delta0)
    node_section = d0_pinv @ edge_section
    exact = delta0 @ node_section

    if delta1 is not None:
        d1T = delta1.T
        d1T_pinv = torch.linalg.pinv(d1T)
        residual = edge_section - exact
        face_coeff = d1T_pinv @ residual
        coexact = d1T @ face_coeff
    else:
        coexact = torch.zeros_like(edge_section)

    harmonic = edge_section - exact - coexact

    total = torch.norm(edge_section).item() ** 2 + 1e-10
    return {
        'exact_fraction': torch.norm(exact).item() ** 2 / total,
        'harmonic_fraction': torch.norm(harmonic).item() ** 2 / total,
        'coexact_fraction': torch.norm(coexact).item() ** 2 / total,
    }
