"""
Edge-set constructors for the cycle-aware H^0 experiments.

A "fact" is anything with multiple paraphrase expressions; we name nodes
``f{fact_id}_p{paraphrase_idx}``. Each constructor returns a list of edges
(pairs of node names) plus the sorted set of node names that appear.

Three topologies:

* ``pair_edges``: chain paraphrases as a forest, b_1 = 0.
* ``ring_edges``: connect all paraphrases of a fact in a cycle, b_1 = 1
  per fact.
* ``clique_edges``: every pair of paraphrases of a fact, b_1 = (K-1)(K-2)/2
  per fact for K paraphrases.
"""

from typing import Iterable, List, Sequence, Tuple

Edge = Tuple[str, str]


def _node_name(fact_id: int, paraphrase_idx: int) -> str:
    return f"f{fact_id}_p{paraphrase_idx}"


def pair_edges(
    facts: Sequence[dict],
    fact_ids: Iterable[int],
    max_chain: int = 3,
) -> Tuple[List[Edge], List[str]]:
    """
    Pair-only chain: each fact contributes ``max_chain`` consecutive edges
    p0-p1, p1-p2, ... The resulting graph is a forest (b_1 = 0).
    """
    edges: List[Edge] = []
    nodes = set()
    for i in fact_ids:
        exprs = facts[i]["expressions"]
        for j in range(min(max_chain, len(exprs) - 1)):
            a = _node_name(i, j)
            b = _node_name(i, j + 1)
            edges.append((a, b))
            nodes.add(a)
            nodes.add(b)
    return edges, sorted(nodes)


def ring_edges(
    facts: Sequence[dict],
    fact_ids: Iterable[int],
    max_paraphrases: int = 5,
) -> Tuple[List[Edge], List[str]]:
    """
    Multi-paraphrase ring: each fact with K paraphrases gets a closed
    K-cycle. b_1 = 1 per fact when K >= 3, else the fact contributes no
    cycles and is skipped.
    """
    edges: List[Edge] = []
    nodes = set()
    for i in fact_ids:
        exprs = facts[i]["expressions"]
        K = min(max_paraphrases, len(exprs))
        if K < 3:
            continue
        ring = [_node_name(i, j) for j in range(K)]
        for j in range(K):
            edges.append((ring[j], ring[(j + 1) % K]))
            nodes.add(ring[j])
    return edges, sorted(nodes)


def clique_edges(
    facts: Sequence[dict],
    fact_ids: Iterable[int],
    max_paraphrases: int = 5,
) -> Tuple[List[Edge], List[str]]:
    """
    Multi-paraphrase clique: every pair of paraphrases of a fact joined.
    b_1 = (K-1)(K-2)/2 per fact when K >= 3.
    """
    edges: List[Edge] = []
    nodes = set()
    for i in fact_ids:
        exprs = facts[i]["expressions"]
        K = min(max_paraphrases, len(exprs))
        if K < 3:
            continue
        names = [_node_name(i, j) for j in range(K)]
        for a in range(K):
            for b in range(a + 1, K):
                edges.append((names[a], names[b]))
            nodes.add(names[a])
    return edges, sorted(nodes)


def count_ring_cycles(facts: Sequence[dict], fact_ids: Iterable[int],
                     max_paraphrases: int = 5) -> int:
    """
    First Betti number of the ring construction.

    Each fact with at least three paraphrases contributes one independent
    cycle; facts with fewer paraphrases are skipped.
    """
    return sum(
        1 for i in fact_ids
        if min(max_paraphrases, len(facts[i]["expressions"])) >= 3
    )


def count_clique_cycles(facts: Sequence[dict], fact_ids: Iterable[int],
                        max_paraphrases: int = 5) -> int:
    """
    First Betti number of the clique construction.

    Each fact with K paraphrases (K >= 3) contributes (K-1)(K-2)/2
    independent cycles; smaller facts are skipped.
    """
    total = 0
    for i in fact_ids:
        K = min(max_paraphrases, len(facts[i]["expressions"]))
        if K >= 3:
            total += (K - 1) * (K - 2) // 2
    return total
