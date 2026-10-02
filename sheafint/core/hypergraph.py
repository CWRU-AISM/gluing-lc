"""
Hyperedge sheaves for transformer interpretability with genuine cellular cohomology.

The paraphrase matching graph used by ScalableSheaf has H^1 = 0 because it
contains no cycles. To obtain genuine
sheaf cohomology with H^1 != 0, we need a richer cellular structure: cycles
without filling, or non-trivial monodromy on filled cycles.

This module bridges semantic-equivalence-class data (paraphrases of one fact,
or facts sharing a relation) into the inputs that ScalableSheaf.fit() accepts:
  pair_overlaps : Dict[(node_i, node_j), Tensor[indices]]
  triple_overlaps : Dict[(node_i, node_j, node_k), Tensor[indices]]

The key construction choice is whether a triple of mutually-connected nodes gets
filled in as a 2-simplex. Filling all triangles makes the complex contractible
and H^1 collapses; leaving filtered triangles unfilled produces genuine 1-cycles
(the Vietoris-Rips construction). We expose both modes plus a hyperedge-cluster
mode that builds richer fact-relation hypergraphs.

References:
    Hansen & Ghrist (2019), "Toward a Spectral Theory of Cellular Sheaves"
    Bodnar et al. (2022), "Neural Sheaf Diffusion"
    Duta et al. (2025), "Directional Sheaf Hypergraph Networks"
"""
from __future__ import annotations

from itertools import combinations
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch


PairKey = Tuple[str, str]
TripleKey = Tuple[str, str, str]


class HyperedgeComplex:
    """A simplicial / hypergraph complex generated from semantic-equivalence clusters.

    Each cluster is a set of nodes (paraphrases of one fact, or facts sharing a
    relation). The complex contains 0-, 1-, and 2-simplices selected by mode:

      mode='clique'   : every cluster of size k contributes the full (k-1)-simplex
                        (filled). H^1 collapses on contractible cliques but cycles
                        formed across clusters can survive.
      mode='matching' : each cluster contributes a perfect matching only (one
                        edge per consecutive pair). Reproduces the paper's
                        matching graph.
      mode='ring'     : each cluster of size k contributes a cycle (k edges, no
                        triangles). H^1 dim = number of independent cycles.
      mode='triangulated' : same simplices as 'clique' (every within-cluster
                        triangle filled); cross-cluster edges added with
                        :meth:`add_edge` stay unfilled.

    Cross-cluster connections are added when the same node appears in multiple
    clusters; this is how relation hypergraphs (multiple facts sharing an
    entity) generate non-trivial topology.
    """

    def __init__(self, mode: str = 'clique'):
        if mode not in {'clique', 'matching', 'ring', 'triangulated'}:
            raise ValueError(f"unknown mode: {mode}")
        self.mode = mode
        self.nodes: List[str] = []
        self._node_set: set[str] = set()
        self.edges: List[PairKey] = []
        self._edge_set: set[PairKey] = set()
        self.faces: List[TripleKey] = []
        self._face_set: set[TripleKey] = set()

    def _add_node(self, n: str) -> None:
        if n not in self._node_set:
            self._node_set.add(n)
            self.nodes.append(n)

    def add_edge(self, a: str, b: str) -> None:
        """Add an (unfilled) edge, e.g. a cross-cluster link between facts."""
        if a == b:
            return
        e = (min(a, b), max(a, b))
        if e not in self._edge_set:
            self._edge_set.add(e)
            self.edges.append(e)
        self._add_node(a)
        self._add_node(b)

    def _add_face(self, a: str, b: str, c: str) -> None:
        if len({a, b, c}) < 3:
            return
        f = tuple(sorted((a, b, c)))
        if f not in self._face_set:
            self._face_set.add(f)
            self.faces.append(f)

    def add_cluster(self, members: Sequence[str]) -> None:
        members = list(dict.fromkeys(members))
        for m in members:
            self._add_node(m)
        if len(members) < 2:
            return
        if self.mode == 'matching':
            for a, b in zip(members[::2], members[1::2]):
                self.add_edge(a, b)
        elif self.mode == 'ring':
            for a, b in zip(members, members[1:] + members[:1]):
                self.add_edge(a, b)
        else:
            for a, b in combinations(members, 2):
                self.add_edge(a, b)
            for a, b, c in combinations(members, 3):
                self._add_face(a, b, c)

    def to_overlaps(
        self,
        feature_index_per_node: Dict[str, torch.Tensor],
    ) -> Tuple[Dict[PairKey, torch.Tensor], Dict[TripleKey, torch.Tensor]]:
        """Produce the pair_overlaps / triple_overlaps dicts ScalableSheaf expects.

        feature_index_per_node[n] gives the row-indices of node n's features
        that participate in the overlap; for a single-pooled-vector-per-node
        sheaf, pass torch.tensor([0]) for every node.
        """
        pair_overlaps: Dict[PairKey, torch.Tensor] = {}
        for a, b in self.edges:
            if a in feature_index_per_node and b in feature_index_per_node:
                idx = feature_index_per_node[a]
                pair_overlaps[(a, b)] = idx if isinstance(idx, torch.Tensor) else torch.tensor(idx)
        triple_overlaps: Dict[TripleKey, torch.Tensor] = {}
        for a, b, c in self.faces:
            keep = a in feature_index_per_node and b in feature_index_per_node and c in feature_index_per_node
            if keep:
                idx = feature_index_per_node[a]
                triple_overlaps[(a, b, c)] = idx if isinstance(idx, torch.Tensor) else torch.tensor(idx)
        return pair_overlaps, triple_overlaps

    def report(self) -> Dict[str, int]:
        return {
            'n_nodes': len(self.nodes),
            'n_edges': len(self.edges),
            'n_faces': len(self.faces),
            'mode': self.mode,
        }


def simplicial_coboundaries(
    node_index: Dict[str, int],
    edges: Sequence[PairKey],
    faces: Sequence[TripleKey],
) -> Tuple[np.ndarray, np.ndarray]:
    """Signed coboundaries of a 2-complex with trivial R coefficients.

    ``d0`` is (E x V) with -1 / +1 at the endpoints of each edge (u, v);
    ``d1`` is (F x E) with boundary([a,b,c]) = [b,c] - [a,c] + [a,b], edges
    matched up to orientation.
    """
    d0 = np.zeros((len(edges), len(node_index)), dtype=np.float64)
    for e_idx, (u, v) in enumerate(edges):
        d0[e_idx, node_index[u]] = -1.0
        d0[e_idx, node_index[v]] = 1.0
    edge_idx = {e: i for i, e in enumerate(edges)}
    d1 = np.zeros((len(faces), len(edges)), dtype=np.float64)
    for f_idx, (a, b, c) in enumerate(faces):
        for (x, y), sign in [((b, c), 1.0), ((a, c), -1.0), ((a, b), 1.0)]:
            e = (min(x, y), max(x, y))
            if e in edge_idx:
                d1[f_idx, edge_idx[e]] = sign
    return d0, d1


def algebraic_cohomology(
    n_nodes: int,
    edges: List[PairKey],
    faces: List[TripleKey],
    node_index: Dict[str, int],
    stalk_dim: int = 1,
) -> Dict[str, int]:
    """Pure-topology cohomology with trivial sheaf (stalks = R^stalk_dim, identity restrictions).

    This isolates the topological contribution: H^k here equals stalk_dim * H^k(complex).
    Useful for verifying that a constructed complex actually has H^1 != 0 before fitting
    a learned sheaf on top of it.
    """
    n_e = len(edges)
    n_f = len(faces)
    d0, d1 = simplicial_coboundaries(node_index, edges, faces)
    rank_d0 = int(np.linalg.matrix_rank(d0)) if d0.size else 0
    rank_d1 = int(np.linalg.matrix_rank(d1)) if d1.size else 0
    H0 = n_nodes - rank_d0
    H1 = max(0, (n_e - rank_d1) - rank_d0)
    H2 = max(0, n_f - rank_d1)
    return {
        'H0': stalk_dim * H0,
        'H1': stalk_dim * H1,
        'H2': stalk_dim * H2,
        'n_nodes': n_nodes,
        'n_edges': n_e,
        'n_faces': n_f,
        'rank_d0': rank_d0,
        'rank_d1': rank_d1,
    }


__all__ = [
    'HyperedgeComplex',
    'algebraic_cohomology',
    'simplicial_coboundaries',
]
