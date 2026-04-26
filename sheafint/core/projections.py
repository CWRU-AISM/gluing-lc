# Learn node projections and edge transport maps for sheaf construction.

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
    # Joint PCA over all overlapping samples enforces transitivity by sharing the projection.
    feature_dims = {n: features[n].shape[1] for n in nodes if n in features}
    unique_dims = set(feature_dims.values())

    if len(unique_dims) != 1:
        return _learn_heterogeneous_projections(
            features, pair_overlaps, nodes, edge_dim, device, use_randomized_svd
        )

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


def _learn_heterogeneous_projections(
    features: Dict[str, torch.Tensor],
    pair_overlaps: Dict[Tuple[str, str], torch.Tensor],
    nodes,
    edge_dim: int,
    device: str,
    use_randomized_svd: bool,
):
    # Per-node projections plus Procrustes transports for varying feature dimensions.
    node_projections: Dict[str, torch.Tensor] = {}
    edge_transports: Dict[Tuple[str, str], torch.Tensor] = {}

    for node, feat in features.items():
        if feat.shape[0] == 0:
            continue

        centered = feat - feat.mean(dim=0, keepdim=True)
        rank = min(edge_dim, feat.shape[1])

        if use_randomized_svd and feat.shape[0] > 100:
            _, _, Vt = randomized_truncated_svd(centered, rank, device=device)
        else:
            _, _, Vt = torch.linalg.svd(centered, full_matrices=False)
            Vt = Vt[:rank]

        if rank < edge_dim:
            padding = torch.zeros(edge_dim - rank, feat.shape[1], device=device)
            Vt = torch.cat([Vt, padding], dim=0)

        node_projections[node] = Vt.T

    identity = torch.eye(edge_dim, device=device)
    for (i, j), indices in pair_overlaps.items():
        if len(indices) < 2:
            edge_transports[(i, j)] = identity.clone()
            continue

        Pi = node_projections[i]
        Pj = node_projections[j]
        Xi = features[i][indices] @ Pi
        Xj = features[j][indices] @ Pj

        cross = Xi.T @ Xj
        U, _, Vt = torch.linalg.svd(cross)
        edge_transports[(i, j)] = U @ Vt

    return node_projections, edge_transports, edge_dim


def learn_procrustes_projections(
    features: Dict[str, torch.Tensor],
    pair_overlaps: Dict[Tuple[str, str], torch.Tensor],
    nodes,
    edge_dim: int,
    device: str,
):
    # Per-node PCA followed by Procrustes alignment between connected nodes.
    node_projections: Dict[str, torch.Tensor] = {}
    edge_transports: Dict[Tuple[str, str], torch.Tensor] = {}

    for node, feat in features.items():
        if feat.shape[0] == 0:
            continue

        centered = feat - feat.mean(dim=0, keepdim=True)
        rank = min(edge_dim, feat.shape[1])

        _, _, Vt = torch.linalg.svd(centered, full_matrices=False)
        Vt = Vt[:rank]

        if rank < edge_dim:
            padding = torch.zeros(edge_dim - rank, feat.shape[1], device=device)
            Vt = torch.cat([Vt, padding], dim=0)

        node_projections[node] = Vt.T

    identity = torch.eye(edge_dim, device=device)
    for (i, j), indices in pair_overlaps.items():
        if len(indices) < 2:
            edge_transports[(i, j)] = identity.clone()
            continue

        Pi = node_projections[i]
        Pj = node_projections[j]
        Xi = features[i][indices] @ Pi
        Xj = features[j][indices] @ Pj

        cross = Xi.T @ Xj
        U, _, Vt = torch.linalg.svd(cross)
        edge_transports[(i, j)] = U @ Vt

    return node_projections, edge_transports
