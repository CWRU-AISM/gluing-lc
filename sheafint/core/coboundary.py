# Coboundary operator construction (delta0 and delta1) with chain-complex exactness.

from typing import Dict, List, Tuple
import torch


def build_delta0(
    nodes: List[str],
    edges: List[Tuple[str, str]],
    edge_transports: Dict[Tuple[str, str], torch.Tensor],
    edge_dim: int,
    device: str,
) -> torch.Tensor:
    # delta0: C^0 -> C^1, mapping node sections to per-edge consistency residuals.
    n_nodes = len(nodes)
    n_edges = len(edges)
    node_to_idx = {n: i for i, n in enumerate(nodes)}

    delta0 = torch.zeros(n_edges * edge_dim, n_nodes * edge_dim, device=device)
    identity = torch.eye(edge_dim, device=device)

    for e_idx, (i, j) in enumerate(edges):
        i_idx = node_to_idx[i]
        j_idx = node_to_idx[j]
        Q = edge_transports.get((i, j), identity)

        row_start = e_idx * edge_dim
        row_end = row_start + edge_dim

        col_i = slice(i_idx * edge_dim, (i_idx + 1) * edge_dim)
        col_j = slice(j_idx * edge_dim, (j_idx + 1) * edge_dim)

        delta0[row_start:row_end, col_i] = -Q
        delta0[row_start:row_end, col_j] = identity

    return delta0


def build_delta1(
    edges: List[Tuple[str, str]],
    faces: List[Tuple[str, str, str]],
    face_projections: Dict[Tuple[str, str, str], torch.Tensor],
    edge_dim: int,
    face_dim: int,
    device: str,
):
    # delta1: C^1 -> C^2 with signed face-boundary structure ensuring delta1 @ delta0 = 0.
    if not faces:
        return None

    edge_to_idx = {e: i for i, e in enumerate(edges)}
    delta1 = torch.zeros(len(faces) * face_dim, len(edges) * edge_dim, device=device)
    default_projection = torch.eye(face_dim, edge_dim, device=device)

    for f_idx, (i, j, k) in enumerate(faces):
        row_start = f_idx * face_dim
        row_end = row_start + face_dim

        face_proj = face_projections.get((i, j, k), default_projection)

        for edge, sign in [((i, j), 1), ((j, k), 1), ((i, k), -1)]:
            if edge in edge_to_idx:
                e_idx = edge_to_idx[edge]
                col = slice(e_idx * edge_dim, (e_idx + 1) * edge_dim)
                delta1[row_start:row_end, col] = sign * face_proj

    return delta1


def exactness_error(delta0: torch.Tensor, delta1) -> float:
    # Frobenius norm of delta1 @ delta0 (should be ~0 for a valid chain complex).
    if delta1 is None:
        return 0.0
    return torch.norm(delta1 @ delta0, 'fro').item()
