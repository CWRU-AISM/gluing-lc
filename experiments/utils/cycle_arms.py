"""
Graph arms for the edge- and node-matched cycle controls.

Every arm draws from training facts with at least MIN_EXPR expressions so
the pool is identical; nodes are f{fact}_p{paraphrase} (plus _l{layer}
on the two-layer grid). build_arms returns {arm: (edges, nodes)} for
the matched, grid and hyper variants.
"""

from collections import defaultdict

import numpy as np

from sheafint import clique_edges, pair_edges, ring_edges

MIN_EXPR = 6  # every arm draws from facts with >= 6 paraphrases so the pool is identical
K = 5  # paraphrases in the ring / grid


def nname(i, j, l=None):
    return f'f{i}_p{j}' if l is None else f'f{i}_p{j}_l{l}'


def _nodes(edges):
    return sorted({n for e in edges for n in e})


def _eligible(facts, idx):
    return [i for i in idx if len(facts[i]['expressions']) >= MIN_EXPR]


def _relations(facts, idx):
    rel = defaultdict(list)
    for i in idx:
        rel[facts[i]['relation_id']].append(i)
    return {r: v for r, v in rel.items() if len(v) >= 3}  # need >= 3 for a ring


def build_grid_pair(facts, idx):
    e = [(nname(i, j, 1), nname(i, j + 1, 1)) for i in _eligible(facts, idx) for j in range(3)]
    return e, _nodes(e)


def build_grid(facts, idx):
    e = []
    for i in _eligible(facts, idx):
        for l in (0, 1):
            for j in range(K - 1):
                e.append((nname(i, j, l), nname(i, j + 1, l)))
        for j in range(K):
            e.append((nname(i, j, 0), nname(i, j, 1)))
    return e, _nodes(e)


def build_grid_tree(facts, idx, n_para=K - 1):
    e = []
    for i in _eligible(facts, idx):
        for l in (0, 1):
            for j in range(n_para):
                e.append((nname(i, j, l), nname(i, j + 1, l)))
        e.append((nname(i, 0, 0), nname(i, 0, 1)))
    return e, _nodes(e)


def build_hypergraph(facts, idx):
    """Per-fact 5-ring + per-relation ring over p0 nodes. b_1 = n_facts + n_relations."""
    elig = _eligible(facts, idx)
    e = [(nname(i, j), nname(i, (j + 1) % K)) for i in elig for j in range(K)]
    nrel = 0
    for members in _relations(facts, elig).values():
        ring = [nname(i, 0) for i in members]
        e += [(ring[j], ring[(j + 1) % len(ring)]) for j in range(len(ring))]
        nrel += 1
    return e, _nodes(e), len(elig) + nrel


def build_hyper_acyclic(facts, idx, n_para):
    """Per-fact path over n_para edges + per-relation path. b_1 = 0."""
    elig = _eligible(facts, idx)
    e = [(nname(i, j), nname(i, j + 1)) for i in elig for j in range(n_para)]
    for members in _relations(facts, elig).values():
        path = [nname(i, 0) for i in members]
        e += list(zip(path, path[1:]))
    return e, _nodes(e)


def build_arms(variant, facts, idx):
    """Return ({arm: (edges, nodes)}, extra payload fields) for a variant."""
    elig = _eligible(facts, idx)
    if variant == 'matched':
        return {
            'pair_only': pair_edges(facts, elig, max_chain=3),
            'spanning_tree': pair_edges(facts, elig, max_chain=4),
            'acyclic_matched': pair_edges(facts, elig, max_chain=5),
            'ring': ring_edges(facts, elig, max_paraphrases=K),
            'clique': clique_edges(facts, elig, max_paraphrases=K),
        }, {}
    if variant == 'grid':
        return {
            'pair_only': build_grid_pair(facts, idx),
            'grid': build_grid(facts, idx),
            'grid_tree': build_grid_tree(facts, idx),
            'grid_tree6': build_grid_tree(facts, idx, n_para=K),
        }, {}
    hyper_e, hyper_n, hyper_b1 = build_hypergraph(facts, idx)
    return {
        'pair_only': pair_edges(facts, elig, max_chain=3),
        'hyper_tree': build_hyper_acyclic(facts, idx, K - 1),
        'hyper_acyclic': build_hyper_acyclic(facts, idx, K),
        'hypergraph': (hyper_e, hyper_n),
    }, {'hypergraph_b1': hyper_b1}


def bootstrap_contrasts(score, bases, contrasts, n_queries: int, n_boot: int, rng) -> dict:
    """Paired-bootstrap 95% CIs (percentage points) of ``score(a) - score(b)`` per contrast.

    ``score(basis, rows)`` scores a query resample; ``contrasts`` lists
    ``(key, arm_a, arm_b)``.
    """
    diffs = {key: [] for key, _, _ in contrasts}
    for _ in range(n_boot):
        rows = rng.integers(0, n_queries, n_queries)
        for key, a, b in contrasts:
            diffs[key].append(score(bases[a], rows) - score(bases[b], rows))
    return {key: [float(np.percentile(d, 2.5) * 100), float(np.percentile(d, 97.5) * 100)]
            for key, d in diffs.items()}
