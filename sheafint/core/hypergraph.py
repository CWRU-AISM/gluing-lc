"""
Hyperedge sheaves for transformer interpretability with genuine cellular cohomology.

The matching-graph construction in scalable_sheaf reduces algebraically to H^1 = 0
(Remark 1 in the paper) because the graph contains no cycles. To obtain genuine
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
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch


SimplexId = Tuple[str, ...]
PairKey = Tuple[str, str]
TripleKey = Tuple[str, str, str]


def _sorted(simplex: Iterable[str]) -> SimplexId:
    return tuple(sorted(set(simplex)))


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
      mode='vr'       : Vietoris-Rips on a precomputed distance matrix; edges
                        included by threshold, triangles only where all three
                        edges exist.
      mode='ring'     : each cluster of size k contributes a cycle (k edges, no
                        triangles). H^1 dim = number of independent cycles.

    Cross-cluster connections are added when the same node appears in multiple
    clusters; this is how relation hypergraphs (multiple facts sharing an
    entity) generate non-trivial topology.
    """

    def __init__(self, mode: str = 'clique'):
        if mode not in {'clique', 'matching', 'vr', 'ring', 'triangulated'}:
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

    def _add_edge(self, a: str, b: str) -> None:
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
                self._add_edge(a, b)
        elif self.mode == 'ring':
            for a, b in zip(members, members[1:] + members[:1]):
                self._add_edge(a, b)
        elif self.mode == 'clique':
            for a, b in combinations(members, 2):
                self._add_edge(a, b)
            for a, b, c in combinations(members, 3):
                self._add_face(a, b, c)
        elif self.mode == 'triangulated':
            # Paraphrases within a cluster form a ring; their pairwise edges and
            # all 3-paraphrase triangles are filled, killing the within-cluster
            # 1-cycle. Cross-cluster edges (added separately) stay unfilled.
            for a, b in combinations(members, 2):
                self._add_edge(a, b)
            for a, b, c in combinations(members, 3):
                self._add_face(a, b, c)
        # 'vr' is added externally via add_vr_edges

    def add_vr_edges(
        self,
        names: Sequence[str],
        distances: np.ndarray,
        edge_quantile: float = 0.2,
        face_threshold: Optional[float] = None,
    ) -> None:
        """Add Vietoris-Rips edges at the given distance quantile, faces only
        when all three edges exist."""
        if distances.shape != (len(names), len(names)):
            raise ValueError("distances must be square and match names length")
        for n in names:
            self._add_node(n)
        flat = distances[np.triu_indices_from(distances, k=1)]
        thresh = float(np.quantile(flat, edge_quantile))
        for i, j in zip(*np.triu_indices_from(distances, k=1)):
            if distances[i, j] <= thresh:
                self._add_edge(names[int(i)], names[int(j)])
        if face_threshold is None:
            face_threshold = thresh
        for a, b, c in combinations(range(len(names)), 3):
            if (distances[a, b] <= face_threshold
                    and distances[b, c] <= face_threshold
                    and distances[a, c] <= face_threshold):
                self._add_face(names[a], names[b], names[c])

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
    # delta0: edges x nodes
    d0 = np.zeros((n_e, n_nodes), dtype=np.float64)
    for e_idx, (u, v) in enumerate(edges):
        d0[e_idx, node_index[u]] = -1.0
        d0[e_idx, node_index[v]] = 1.0
    # delta1: faces x edges with alternating signs (signed boundary of triangle)
    edge_idx = {e: i for i, e in enumerate(edges)}
    d1 = np.zeros((n_f, n_e), dtype=np.float64)
    for f_idx, (a, b, c) in enumerate(faces):
        # boundary of [a,b,c] = [b,c] - [a,c] + [a,b]
        for edge, sign in [((b, c), 1), ((a, c), -1), ((a, b), 1)]:
            e = (min(*edge), max(*edge))
            if e in edge_idx:
                d1[f_idx, edge_idx[e]] = sign
    rank_d0 = int(np.linalg.matrix_rank(d0)) if d0.size else 0
    rank_d1 = int(np.linalg.matrix_rank(d1)) if d1.size else 0
    H0 = n_nodes - rank_d0
    # H1 = ker(d1) - im(d0); ker(d1) = n_e - rank(d1); im(d0) = rank(d0)
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


def build_relation_hypergraph_from_counterfact(
    facts: List[Dict],
    paraphrases_per_fact: int = 5,
    cross_relation: bool = True,
    fill_triangles: bool = False,
) -> HyperedgeComplex:
    """Construct a relation hypergraph from CounterFact-style records.

    Each fact contributes one cluster of paraphrase nodes (named "fact{i}_p{j}").
    When cross_relation is True, facts sharing the same relation_id are connected
    via cross-cluster edges; this is what creates 1-cycles when relations
    overlap in non-tree-like patterns.

    fill_triangles=True uses 'triangulated' mode: within-fact paraphrase triangles
    are filled (paraphrase rings become contractible), so H^1 captures only the
    cross-relation cycles. fill_triangles=False uses 'ring' mode: paraphrase rings
    stay non-contractible.
    """
    mode = 'triangulated' if fill_triangles else 'ring'
    H = HyperedgeComplex(mode=mode)
    relation_to_facts: Dict[str, List[int]] = {}
    for i, fact in enumerate(facts):
        members = [f"fact{i}_p{j}" for j in range(min(paraphrases_per_fact, len(fact.get('paraphrases', []))))]
        H.add_cluster(members)
        rel = fact.get('relation_id') or fact.get('relation')
        if cross_relation and rel:
            relation_to_facts.setdefault(str(rel), []).append(i)
    if cross_relation:
        for rel, fact_ids in relation_to_facts.items():
            if len(fact_ids) < 2:
                continue
            # Connect first paraphrase of each fact within this relation as a ring
            ring = [f"fact{i}_p0" for i in fact_ids]
            for a, b in zip(ring, ring[1:] + ring[:1]):
                H._add_edge(a, b)
    return H


__all__ = [
    'HyperedgeComplex',
    'algebraic_cohomology',
    'build_relation_hypergraph_from_counterfact',
]
