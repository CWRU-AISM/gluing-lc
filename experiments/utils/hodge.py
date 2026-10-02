"""
Shared helpers for the Hodge harmonic-mass experiments: the multi-layer
matching grid (Table 1, appendix scale-up), the relation hypergraph
(appendix A.1), and their triangulated 2-complex variants (appendix A.2).

All transports are the identity, so the lifted coboundaries are
``delta0 = d0_graph kron I_k`` and ``delta1 = d1_graph kron I_k`` and their
pseudo-inverses lift likewise. The Hodge decomposition therefore acts on
``(n_edges, k)`` arrays and never materialises the lifted operators, which
is what makes k = 64 / 128 tractable.

Activations are extracted in bf16 without quantization, one prompt per
forward pass (no padding), truncated to 64 tokens.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Sequence

import numpy as np
import torch

from sheafint.core.cohomology import hodge_decomposition
from sheafint.core.holonomy import procrustes
from sheafint.core.hypergraph import HyperedgeComplex, algebraic_cohomology
from sheafint.core.sheaf import ScalableSheaf

from .datasets import load_mrpc_pairs
from .model_io import load_causal_model


def load_model(model_name: str, device: str | None = None):
    """Unquantized bf16 model on a single device; returns ``(model, tokenizer, device)``."""
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model, tok = load_causal_model(model_name, quantize='none', dtype=torch.bfloat16, device_map=device)
    return model, tok, device


@torch.no_grad()
def extract_layers(model, tokenizer, prompts: Sequence[str], layers: Sequence[int],
                   device: str) -> Dict[int, torch.Tensor]:
    """dict[layer] -> (N, d) mean-pooled hidden states (float32, CPU)."""
    out = {l: [] for l in layers}
    for p in prompts:
        ids = tokenizer(p, return_tensors='pt', truncation=True, max_length=64).to(device)
        h_all = model(**ids, output_hidden_states=True).hidden_states
        mask = ids.get('attention_mask', torch.ones_like(ids['input_ids']))
        m = mask.unsqueeze(-1).float()
        for l in layers:
            pooled = (h_all[l] * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)
            out[l].append(pooled.squeeze(0).float().cpu())
    return {l: torch.stack(out[l]) for l in layers}


def grid_layer_features(model_name: str, device: str | None, n_pairs: int, layer_counts: Sequence[int]):
    """Activations of the 2 * n_pairs MRPC train sentences at every layer the subsets use.

    Returns ``(subsets, {layer: (2N, d)}, n_layers_total, d_model)``; layer
    counts larger than the model depth are dropped.
    """
    sentences = [s for a, b in load_mrpc_pairs(n_pairs, split='train') for s in (a, b)]
    model, tok, device = load_model(model_name, device)
    n_layers_total = model.config.num_hidden_layers + 1
    subsets = [s for s in (choose_layers(n_layers_total, L) for L in layer_counts) if s is not None]
    all_layers = sorted({l for s in subsets for l in s})
    features = extract_layers(model, tok, sentences, all_layers, device)
    del model
    torch.cuda.empty_cache()
    return subsets, features, n_layers_total, features[all_layers[0]].shape[1]


def choose_layers(n_layers_total: int, L: int) -> List[int] | None:
    """Layer subset of size L: L=1 -> the 2/3-depth layer; L > depth -> None."""
    if L == 1:
        return [int(n_layers_total * 2 // 3)]
    if L > n_layers_total:
        return None
    return np.linspace(2, n_layers_total - 1, L, dtype=int).tolist()


def build_grid(n_pairs: int, L: int, diagonals: bool = False):
    """Vertices (i, side, l_pos); horizontal, vertical and (optionally)
    diagonal edges; two triangles per square when ``diagonals`` is set.

    Returns ``(node_idx, horiz, vert, diag, faces)``. Every edge is stored
    as ``(u, v)`` with ``u < v`` so the face boundary signs need no flip.
    """
    node_idx = {}
    for i in range(n_pairs):
        for s in (0, 1):
            for l_pos in range(L):
                node_idx[(i, s, l_pos)] = len(node_idx)
    horiz = [((i, 0, l), (i, 1, l)) for i in range(n_pairs) for l in range(L)]
    vert = [((i, s, l), (i, s, l + 1))
            for i in range(n_pairs) for s in (0, 1) for l in range(L - 1)]
    diag, faces = [], []
    if diagonals:
        for i in range(n_pairs):
            for l in range(L - 1):
                diag.append(((i, 0, l), (i, 1, l + 1)))
                faces.append(((i, 0, l), (i, 1, l), (i, 1, l + 1)))      # T1
                faces.append(((i, 0, l), (i, 0, l + 1), (i, 1, l + 1)))  # T2
    return node_idx, horiz, vert, diag, faces


def fit_layer_projections(layer_features: Dict[int, torch.Tensor], edge_dim: int):
    """Per-layer P_l: top-k right singular vectors of the centred activations."""
    P = {}
    for l, feats in layer_features.items():
        Xc = feats - feats.mean(dim=0, keepdim=True)
        _, _, Vt = torch.linalg.svd(Xc, full_matrices=False)
        k = min(edge_dim, Vt.shape[0])
        P[l] = Vt[:k].T.contiguous()
    return P


def grid_cochain(horiz, vert, diag, layer_features, layer_subset, P, k) -> np.ndarray:
    """Model-derived edge cochain alpha as an (n_edges, k) array.

    horizontal: P_l^T (h_b - h_a)
    vertical:   (P_{l+1}^T h_s^{l+1} - mu_{l+1}) - (P_l^T h_s^l - mu_l) R_{l->l+1}
    diagonal:   same as vertical but from h_a^l to h_b^{l+1}
    R is fit by Procrustes on the centred projected activations of all 2N
    sentences; the sentence index of node (i, side, .) is 2i + side.
    """
    Q = {}
    for l_pos in range(len(layer_subset) - 1):
        l, l_next = layer_subset[l_pos], layer_subset[l_pos + 1]
        proj_l = layer_features[l] @ P[l][:, :k]
        proj_n = layer_features[l_next] @ P[l_next][:, :k]
        R = procrustes(proj_l - proj_l.mean(dim=0, keepdim=True),
                           proj_n - proj_n.mean(dim=0, keepdim=True))
        Q[l_pos] = (R, proj_l.mean(dim=0), proj_n.mean(dim=0))

    def transported_diff(u, v):
        (i_u, s_u, l_u), (i_v, s_v, l_v) = u, v
        l, l_next = layer_subset[l_u], layer_subset[l_v]
        R, mu_l, mu_n = Q[l_u]
        h_l = layer_features[l][2 * i_u + s_u] @ P[l][:, :k]
        h_n = layer_features[l_next][2 * i_v + s_v] @ P[l_next][:, :k]
        return (h_n - mu_n) - (h_l - mu_l) @ R

    alpha = np.zeros((len(horiz) + len(vert) + len(diag), k), dtype=np.float64)
    for ei, (u, v) in enumerate(horiz):
        l = layer_subset[u[2]]
        feats = layer_features[l]
        alpha[ei] = ((feats[2 * v[0] + v[1]] - feats[2 * u[0] + u[1]]) @ P[l][:, :k]).numpy()
    for ei, (u, v) in enumerate(vert + diag):
        alpha[len(horiz) + ei] = transported_diff(u, v).numpy()
    return alpha


def build_fact_complex(facts, n_per_fact: int, mode: str,
                       relation_rings: bool) -> HyperedgeComplex:
    """One HyperedgeComplex over nodes ``f{i}_p{j}``: each fact's paraphrases
    form a cluster in ``mode``; with ``relation_rings`` the first paraphrases
    of the facts sharing a relation are chained in an unfilled ring."""
    H = HyperedgeComplex(mode=mode)
    rel_to_facts = defaultdict(list)
    for i, fact in enumerate(facts):
        n = min(n_per_fact, len(fact['paraphrases']))
        H.add_cluster([f'f{i}_p{j}' for j in range(n)])
        rel_to_facts[fact['relation_id']].append(i)
    if relation_rings:
        for fact_ids in rel_to_facts.values():
            if len(fact_ids) < 2:
                continue
            ring = [f'f{i}_p0' for i in fact_ids]
            for a, b in zip(ring, ring[1:] + ring[:1]):
                H.add_edge(a, b)
    return H


def fact_prompts(facts, n_per_fact: int) -> Dict[str, str]:
    """``{f{i}_p{j}: paraphrase}`` for the first ``n_per_fact`` paraphrases of every fact."""
    return {f'f{i}_p{j}': fact['paraphrases'][j]
            for i, fact in enumerate(facts) for j in range(min(n_per_fact, len(fact['paraphrases'])))}


def fit_fact_sheaf(features: torch.Tensor, names, complex_: HyperedgeComplex,
                   edge_dim: int, face_dim: int, device: str) -> Dict:
    """Fit a joint-PCA ScalableSheaf on one pooled vector per node of ``complex_``.

    Reports the algebraic cohomology, the Laplacian spectrum, and the Hodge
    split of one random unit edge cochain.
    """
    feature_dict = {name: features[i:i + 1].to(device) for i, name in enumerate(names)}
    pair_overlaps, triple_overlaps = complex_.to_overlaps({n: torch.tensor([0]) for n in names})
    sheaf = ScalableSheaf(edge_dim=edge_dim, face_dim=face_dim, device=device)
    sheaf.fit(feature_dict, pair_overlaps, triple_overlaps or None, method='joint_pca')
    cohom = sheaf.compute_cohomology()
    spectrum = sheaf.compute_laplacian_spectrum(k=min(50, sheaf.delta0.shape[1]))
    if sheaf.edges and sheaf.delta0 is not None:
        probe = torch.randn(len(sheaf.edges) * sheaf.edge_dim, device=device)
        hodge = hodge_decomposition(probe / (probe.norm() + 1e-8), sheaf.delta0, sheaf.delta1)
    else:
        hodge = {'exact_fraction': 0.0, 'harmonic_fraction': 0.0, 'coexact_fraction': 0.0}
    return {
        'edge_dim': sheaf.edge_dim,
        'n_nodes': len(sheaf.nodes), 'n_edges': len(sheaf.edges), 'n_faces': len(sheaf.faces),
        **{key: cohom[key] for key in ('H0_dim', 'H1_dim', 'rank_delta0', 'rank_delta1', 'dim_C0', 'dim_C1')},
        'spectral_h0': spectrum['h0_dim'],
        'spectral_gap': spectrum.get('spectral_gap', float('nan')),
        'hodge_random_probe': hodge,
    }


def betti(node_idx, edges, faces) -> Dict[str, int]:
    """b_0, b_1, b_2 with trivial R-coefficients via ``algebraic_cohomology``."""
    c = algebraic_cohomology(len(node_idx), edges, faces, node_idx, stalk_dim=1)
    return {'V': c['n_nodes'], 'E': c['n_edges'], 'F': c['n_faces'],
            'b_0': c['H0'], 'b_1': c['H1'], 'b_2': c['H2'],
            'rank_delta0': c['rank_d0'], 'rank_delta1': c['rank_d1']}


def hodge_2d(alpha_2d: np.ndarray, d0_graph: np.ndarray, d1_graph: np.ndarray | None = None):
    """alpha = exact + harmonic + coexact on (n_edges, k) arrays using the
    Kronecker structure of the lifted operators."""
    # lstsq, not pinv: NumPy 2.5's pinv returns wrong projections on these rank-deficient matrices.
    exact = d0_graph @ np.linalg.lstsq(d0_graph, alpha_2d, rcond=None)[0]
    residual = alpha_2d - exact
    if d1_graph is not None and d1_graph.shape[0] > 0:
        d1T = d1_graph.T
        coexact = d1T @ np.linalg.lstsq(d1T, residual, rcond=None)[0]
    else:
        coexact = np.zeros_like(alpha_2d)
    return exact, residual - coexact, coexact


def hodge_random(d0_graph, d1_graph, k: int, n_probes: int = 10, seed: int = 0) -> float:
    """Mean harmonic energy fraction of ``n_probes`` random unit edge cochains."""
    rng = np.random.default_rng(seed)
    if d0_graph.shape[0] == 0:
        return 0.0
    fractions = []
    for _ in range(n_probes):
        x = rng.standard_normal((d0_graph.shape[0], k))
        x = x / (np.linalg.norm(x) + 1e-12)
        _, harm, _ = hodge_2d(x, d0_graph, d1_graph)
        fractions.append(float((harm ** 2).sum()))
    return float(np.mean(fractions))


def sq(x: np.ndarray) -> float:
    return float((x ** 2).sum())

