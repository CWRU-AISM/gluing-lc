"""
Cycle holonomy on heterogeneous-Q sheaves.

Given per-node activations on a graph, fit a per-node PCA P_v of fixed
edge dimension k and a per-edge Procrustes transport Q_ij. Walking the
transports around a closed cycle gives a holonomy operator H_gamma whose
distance from the identity (Frobenius) measures the obstruction to gluing.

Helpers:

* ``fit_per_node_pca``: per-node PCA with random padding for rank-deficient
  nodes (small token counts).
* ``procrustes``: orthogonal R minimising ||X R - Y||_F.
* ``fit_procrustes_transports``: per-edge orthogonal Procrustes between
  projected node features.
* ``find_fundamental_cycles``: BFS / DFS-based fundamental cycle basis.
* ``cycle_holonomy``, ``cycle_obstructions``: compose transports and report
  ||I - H_gamma||_F per cycle.
"""

from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

import torch

NodeId = str
Edge = Tuple[NodeId, NodeId]


def fit_per_node_pca(
    node_features: Dict[NodeId, torch.Tensor],
    edge_dim: int,
    device: torch.device,
    generator: torch.Generator,
) -> Dict[NodeId, torch.Tensor]:
    """
    Per-node PCA padded to ``edge_dim`` columns. If a node has fewer than
    ``edge_dim`` non-zero singular values (small N), pad with random
    orthogonal directions in the orthogonal complement.
    """
    out: Dict[NodeId, torch.Tensor] = {}
    if not node_features:
        return out
    d = next(iter(node_features.values())).shape[1]
    for name, feat in node_features.items():
        centered = feat - feat.mean(dim=0, keepdim=True)
        _, _, Vt = torch.linalg.svd(centered, full_matrices=False)
        avail = Vt.shape[0]
        if avail >= edge_dim:
            P = Vt[:edge_dim].T
        else:
            extra = torch.randn(edge_dim - avail, d, device=device,
                                generator=generator)
            extra = extra - extra @ Vt.T @ Vt
            extra, _ = torch.linalg.qr(extra.T)
            P = torch.cat([Vt.T, extra[:, :edge_dim - avail]], dim=1)
        P, _ = torch.linalg.qr(P)
        out[name] = P[:, :edge_dim]
    return out


def procrustes(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Orthogonal R minimising ||X R - Y||_F."""
    U, _, Vt = torch.linalg.svd(X.T @ Y, full_matrices=False)
    return U @ Vt


def fit_procrustes_transports(
    node_features: Dict[NodeId, torch.Tensor],
    node_projections: Dict[NodeId, torch.Tensor],
    edges: Sequence[Edge],
    device: torch.device,
    n_overlap_tokens: int = None,
) -> Dict[Edge, torch.Tensor]:
    """
    For each edge (u, v), compute the orthogonal Procrustes transport
    between projected node features. Uses up to ``n_overlap_tokens`` per
    node when given, otherwise the smaller of the two sample counts.
    """
    transports: Dict[Edge, torch.Tensor] = {}
    for (a, b) in edges:
        if a not in node_features or b not in node_features:
            continue
        n = min(node_features[a].shape[0], node_features[b].shape[0])
        if n_overlap_tokens is not None:
            n = min(n, n_overlap_tokens)
        Pa = node_projections[a]
        Pb = node_projections[b]
        Xa = node_features[a][:n].to(device) @ Pa
        Xb = node_features[b][:n].to(device) @ Pb
        transports[(a, b)] = procrustes(Xa, Xb)
    return transports


def find_fundamental_cycles(
    nodes: Sequence[NodeId],
    edges: Sequence[Edge],
    max_len: int = 12,
) -> List[List[NodeId]]:
    """
    Fundamental cycle basis via DFS spanning tree + non-tree edges.
    Cycles are returned as ordered node sequences with the closing edge
    implicit.
    """
    adj: Dict[NodeId, set] = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    parent: Dict[NodeId, NodeId] = {}
    visited: set = set()
    tree_edges: set = set()
    if not nodes:
        return []

    stack = [(nodes[0], None)]
    while stack:
        node, par = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        parent[node] = par
        if par is not None:
            tree_edges.add((min(node, par), max(node, par)))
        for nb in adj[node]:
            if nb not in visited:
                stack.append((nb, node))

    cycles: List[List[NodeId]] = []
    for u, v in edges:
        e = (min(u, v), max(u, v))
        if e in tree_edges:
            continue
        path_u: List[NodeId] = []
        n = u
        while n is not None:
            path_u.append(n)
            n = parent.get(n)
        anc = set(path_u)
        path_v: List[NodeId] = []
        n = v
        while n is not None and n not in anc:
            path_v.append(n)
            n = parent.get(n)
        if n is None:
            continue
        idx = path_u.index(n)
        cycle = path_u[:idx + 1] + list(reversed(path_v))
        if 3 <= len(cycle) <= max_len:
            cycles.append(cycle)
    return cycles


def cycle_holonomy(
    cycle: Sequence[NodeId],
    edge_transports: Dict[Edge, torch.Tensor],
    edge_dim: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Compose Q transports along a closed cycle. Returns the identity if any
    edge along the cycle is missing from ``edge_transports``.
    """
    H = torch.eye(edge_dim, device=device)
    for a, b in zip(cycle, list(cycle[1:]) + [cycle[0]]):
        e = (min(a, b), max(a, b))
        if e in edge_transports:
            Q = edge_transports[e]
            if (a, b) != e:
                Q = Q.T
            H = Q @ H
        else:
            return torch.eye(edge_dim, device=device)
    return H


def cycle_obstructions(
    cycles: Sequence[Sequence[NodeId]],
    edge_transports: Dict[Edge, torch.Tensor],
    edge_dim: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Frobenius distance from identity per cycle.

    For each cycle gamma, compose the Q transports and return
    ||I - H_gamma||_F as a CPU 1-D tensor of length len(cycles).
    """
    eye = torch.eye(edge_dim, device=device)
    out = []
    for cycle in cycles:
        H = cycle_holonomy(cycle, edge_transports, edge_dim, device)
        out.append(torch.linalg.norm(H - eye, ord='fro').item())
    return torch.tensor(out)
