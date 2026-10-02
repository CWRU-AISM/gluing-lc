"""
Node projections and edge transport maps for sheaf construction.

Provides the joint-PCA projection shared across all nodes.
"""

from typing import Dict, Tuple
import torch
from .svd import randomized_truncated_svd


def learn_joint_projections(
    features: Dict[str, torch.Tensor],
    pair_overlaps: Dict[Tuple[str, str], torch.Tensor],
    nodes,
    edge_dim: int,
    device: str,
    use_randomized_svd: bool = True,
):
    """Joint PCA over all overlapping samples enforces transitivity by sharing the projection."""
    feature_dims = {n: features[n].shape[1] for n in nodes if n in features}
    unique_dims = set(feature_dims.values())

    assert len(unique_dims) == 1, f"all nodes must share one feature dim, got {unique_dims}"

    overlap_features = []
    for (i, j), indices in pair_overlaps.items():
        if len(indices) > 0:
            overlap_features.append(features[i][indices])
            overlap_features.append(features[j][indices])

    node_projections: Dict[str, torch.Tensor] = {}
    edge_transports: Dict[Tuple[str, str], torch.Tensor] = {}

    if not overlap_features:
        return node_projections, edge_transports, edge_dim

    stacked = torch.cat(overlap_features, dim=0)
    centered = stacked - stacked.mean(dim=0, keepdim=True)

    actual_edge_dim = min(edge_dim, stacked.shape[0], stacked.shape[1])

    if use_randomized_svd and stacked.shape[0] > 100:
        _, _, Vt = randomized_truncated_svd(centered, actual_edge_dim, device=device)
    else:
        _, _, Vt = torch.linalg.svd(centered, full_matrices=False)
        Vt = Vt[:actual_edge_dim]

    shared_projection = Vt.T

    for node in nodes:
        if node in feature_dims:
            node_projections[node] = shared_projection.clone()

    identity = torch.eye(actual_edge_dim, device=device)
    for edge in pair_overlaps.keys():
        edge_transports[edge] = identity.clone()

    return node_projections, edge_transports, actual_edge_dim

