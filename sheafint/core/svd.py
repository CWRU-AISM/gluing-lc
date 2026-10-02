"""
Randomized SVD utilities used throughout the sheaf pipeline.

Provides truncated SVD with oversampling and a rank-finder used to compute
``dim ker`` for the coboundary operators.
"""

from typing import Tuple
import warnings
import torch
from sklearn.utils.extmath import randomized_svd as _sk_randomized_svd


def randomized_truncated_svd(
    matrix: torch.Tensor,
    rank: int,
    device: str = 'cuda',
    random_state: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Truncated SVD via scikit-learn's randomized solver. Falls back to torch.linalg.svd on failure."""
    rank = min(rank, min(matrix.shape) - 1)

    if rank <= 0:
        empty_u = torch.zeros(matrix.shape[0], 0, device=device)
        empty_s = torch.zeros(0, device=device)
        empty_vt = torch.zeros(0, matrix.shape[1], device=device)
        return empty_u, empty_s, empty_vt

    matrix_np = matrix.cpu().numpy()
    try:
        U, S, Vt = _sk_randomized_svd(matrix_np, n_components=rank, random_state=random_state)
        return (
            torch.tensor(U, device=device, dtype=torch.float32),
            torch.tensor(S, device=device, dtype=torch.float32),
            torch.tensor(Vt, device=device, dtype=torch.float32),
        )
    except Exception as exc:
        warnings.warn(f"Randomized SVD failed: {exc}. Falling back to full SVD.")
        U, S, Vt = torch.linalg.svd(matrix, full_matrices=False)
        return U[:, :rank], S[:rank], Vt[:rank]


def numerical_rank(
    matrix: torch.Tensor,
    rank_cap: int,
    tol: float,
    use_randomized: bool = True,
    device: str = 'cuda',
) -> int:
    """Estimate the numerical rank of a matrix above the given tolerance."""
    if matrix.numel() == 0:
        return 0

    rank = min(rank_cap, min(matrix.shape) - 1)
    if rank <= 0:
        return 0

    if use_randomized and min(matrix.shape) > 100:
        _, singular_values, _ = randomized_truncated_svd(matrix, rank, device=device)
    else:
        _, singular_values, _ = torch.linalg.svd(matrix, full_matrices=False)
        singular_values = singular_values[:rank]

    return int((singular_values > tol).sum().item())
