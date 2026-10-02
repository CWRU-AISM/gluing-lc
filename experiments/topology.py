"""
Cellular cohomology, Hodge harmonic mass, and holonomy of model activations.

    python experiments/topology.py <experiment> --model <hf id> [flags]

All transports are the identity except in holonomy_null, so the Hodge
decomposition runs on (n_edges, k) arrays through the Kronecker structure of
the lifted coboundaries. Models load in bf16 without quantization.
"""

import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np
import torch
import tyro

from sheafint import cycle_obstructions, find_fundamental_cycles
from sheafint.core.h0 import fit_h0_pca, joint_pca_projection
from sheafint.core.hypergraph import algebraic_cohomology, simplicial_coboundaries
from sheafint.data import build_facts
from utils.hodge import (
    betti, build_fact_complex, build_grid, extract_layers, fact_prompts, fit_fact_sheaf, fit_layer_projections,
    grid_cochain, grid_layer_features, hodge_2d, hodge_random, load_model, sq,
)
from utils.holonomy import holonomy_obstructions, mean_std, shuffle_within_relation, token_activations
from utils.model_io import load_causal_model
from utils.results import results_path, write_json


def _fraction(part: float, total: float) -> float:
    return part / total if total > 0 else 0.0


@dataclass
class Grid:
    """Table 1: Hodge harmonic mass of the multi-layer matching grid (1-skeleton).

    Vertices are (sentence, layer) for both sentences of --n_pairs MRPC pairs
    over a layer subset of size L (L = 1 is the 2L/3 layer, larger L are
    spread by linspace(2, depth - 1, L)); b_1 = N(L - 1). Horizontal cochains
    are P_l^T (h_b - h_a), vertical ones the Procrustes-transported residual
    across adjacent layers. With no 2-faces, the harmonic part is
    alpha - d0 d0^+ alpha. --edge_dim 64 and --edge_dim 128 --n_pairs 80 give
    the appendix scale-up.
    """

    model: str = 'gpt2'
    n_pairs: int = 30
    edge_dim: int = 8
    sign_trials: int = 0
    """Also report the harmonic fraction under this many random sign flips of the PCA axes."""
    layer_subsets: Tuple[int, ...] = (1, 2, 4, 8, 16)
    """Layer counts L to test."""
    output_dir: str = 'outputs/multilayer_hodge'
    device: Optional[str] = None

    def run(self):
        out_path = results_path(self.output_dir, self.model, tag=f'k{self.edge_dim}')
        subsets, features, n_layers_total, d_model = grid_layer_features(
            self.model, self.device, self.n_pairs, self.layer_subsets)
        results = []
        for layer_subset in subsets:
            node_idx, horiz, vert, _, _ = build_grid(self.n_pairs, len(layer_subset))
            topo = betti(node_idx, horiz + vert, [])
            del topo['rank_delta1']
            n_h = len(horiz)
            feats = {l: features[l] for l in layer_subset}
            P = fit_layer_projections(feats, self.edge_dim)
            k = min(p.shape[1] for p in P.values())
            alpha = grid_cochain(horiz, vert, [], feats, layer_subset, P, k)
            d0, _ = simplicial_coboundaries(node_idx, horiz + vert, [])
            exact, harmonic, _ = hodge_2d(alpha, d0)

            rng, flips = np.random.default_rng(0), []
            for _ in range(self.sign_trials):
                Pf = {l: p * torch.tensor(rng.choice([-1.0, 1.0], size=p.shape[1]), dtype=p.dtype)
                      for l, p in P.items()}
                af = grid_cochain(horiz, vert, [], feats, layer_subset, Pf, k)
                flips.append(sq(hodge_2d(af, d0)[1]) / sq(af))
            a_h, a_v = sq(alpha[:n_h]), sq(alpha[n_h:])
            h_h, h_v = sq(harmonic[:n_h]), sq(harmonic[n_h:])
            block = {
                'alpha_norm2_total': a_h + a_v, 'alpha_norm2_horiz': a_h, 'alpha_norm2_vert': a_v,
                'harmonic_norm2_total': h_h + h_v, 'harmonic_norm2_horiz': h_h, 'harmonic_norm2_vert': h_v,
                'harmonic_fraction_total': _fraction(h_h + h_v, a_h + a_v),
                'harmonic_fraction_horiz': _fraction(h_h, a_h), 'harmonic_fraction_vert': _fraction(h_v, a_v),
            }
            if flips:
                block['sign_flip_harmonic_fraction'] = {'min': float(np.min(flips)), 'median': float(np.median(flips)),
                                                         'max': float(np.max(flips)), 'n': len(flips)}
            results.append({'n_layers': len(layer_subset), 'layer_indices': layer_subset, 'edge_dim_k': k,
                            'topology': topo, 'n_h': n_h, 'n_v': len(vert), 'block': block,
                            'reconstruction_error': float(np.linalg.norm(alpha - exact - harmonic))})
        write_json(out_path, {'model': self.model, 'n_pairs': self.n_pairs, 'edge_dim': self.edge_dim,
                              'n_layers_total': n_layers_total, 'd_model': d_model, 'results': results})


@dataclass
class Hypergraph:
    """Appendix A.1: genuine H^1 on the relation hypergraph of the 60 templated facts.

    Three complexes on the same paraphrase nodes: matching (H^1 = 0), filled
    fact cliques (H^1 = 0, H^2 = 240), and paraphrase rings plus relation
    rings (H^1 = 66). Reports trivial-sheaf Betti numbers and, for a joint-PCA
    ScalableSheaf on the layer activations, the algebraic cohomology and the
    Hodge split of one random unit edge cochain (unseeded). The paper runs
    the 7B models with --layer 16.
    """

    model: str = 'gpt2'
    paraphrases_per_fact: int = 5
    n_relations: int = 6
    layer: int = 8
    edge_dim: int = 16
    face_dim: int = 8
    output_dir: str = 'outputs/hypergraph_h1'
    device: Optional[str] = None

    def run(self):
        out_path = results_path(self.output_dir, self.model)
        n_par = self.paraphrases_per_fact
        facts = build_facts(self.n_relations, n_par)
        topologies = {'matching': build_fact_complex(facts, n_par, 'matching', False),
                      'clique': build_fact_complex(facts, n_par, 'clique', False),
                      'relation_ring': build_fact_complex(facts, n_par, 'ring', True)}
        topo_results = {}
        for name, c in topologies.items():
            coh = algebraic_cohomology(len(c.nodes), c.edges, c.faces, {n: i for i, n in enumerate(c.nodes)})
            topo_results[name] = {**coh, 'topology': name, **c.report()}

        model, tok, device = load_model(self.model, self.device)
        names = topologies['clique'].nodes
        prompts = fact_prompts(facts, n_par)
        feats = extract_layers(model, tok, [prompts[n] for n in names], [self.layer], device)[self.layer]
        sheaf_results = {name: fit_fact_sheaf(feats[[names.index(n) for n in c.nodes]], c.nodes, c,
                                              self.edge_dim, self.face_dim, device)
                         for name, c in topologies.items()}
        write_json(out_path, {
            'model': self.model, 'n_facts': len(facts), 'n_relations': self.n_relations,
            'paraphrases_per_fact': n_par, 'layer': self.layer, 'edge_dim': self.edge_dim,
            'face_dim': self.face_dim, 'topology_alone': topo_results, 'sheaf_fitted': sheaf_results,
        })


@dataclass
class Triangulated:
    """Appendix A.2: triangulated 2-complex variants (edge dim 64).

    With --complex grid, every square of the multi-layer grid is filled with
    two triangles along the (a, l)-(b, l+1) diagonal, so b_1 = b_2 = 0 and the
    model cochain has no harmonic part; reports the coexact fraction and
    ||delta1 alpha||^2. --complex hypergraph compares matching / ring /
    triangulated relation hypergraphs (each fact's 5-clique filled), all with
    relation rings: Betti numbers, the mean harmonic mass of --n_probes random
    unit cochains, and the shared-PCA model cochain.
    """

    complex: Literal['grid', 'hypergraph'] = 'grid'
    model: str = 'gpt2'
    edge_dim: int = 64
    n_pairs: int = 30
    """grid: MRPC pairs."""
    layer_subsets: Tuple[int, ...] = (2, 4, 8)
    """grid: layer counts L."""
    layer: int = 8
    """hypergraph: extraction layer."""
    n_per_fact: int = 5
    n_relations: int = 6
    n_probes: int = 10
    """hypergraph: random unit cochains."""
    output_dir: Optional[str] = None
    """Default outputs/multilayer_triangulated or outputs/hypergraph_triangulated."""
    device: Optional[str] = None

    def run(self):
        default_dir = f'outputs/{"multilayer" if self.complex == "grid" else "hypergraph"}_triangulated'
        out_path = results_path(self.output_dir or default_dir, self.model, tag=f'k{self.edge_dim}')
        write_json(out_path, self._grid() if self.complex == 'grid' else self._hypergraph())

    def _grid(self):
        subsets, features, n_layers_total, d_model = grid_layer_features(
            self.model, self.device, self.n_pairs, self.layer_subsets)
        results = []
        for layer_subset in subsets:
            node_idx, horiz, vert, diag, faces = build_grid(self.n_pairs, len(layer_subset), diagonals=True)
            edges = horiz + vert + diag
            n_h, n_w, n_d, n_f = len(horiz), len(vert), len(diag), len(faces)
            feats = {l: features[l] for l in layer_subset}
            P = fit_layer_projections(feats, self.edge_dim)
            k = min(p.shape[1] for p in P.values())
            alpha = grid_cochain(horiz, vert, diag, feats, layer_subset, P, k)
            d0, d1 = simplicial_coboundaries(node_idx, edges, faces)
            exact, harmonic, coexact = hodge_2d(alpha, d0, d1)
            beta_norm2 = sq(d1 @ alpha) if n_f else 0.0
            parts = {'horiz': slice(0, n_h), 'vert': slice(n_h, n_h + n_w), 'diag': slice(n_h + n_w, None)}
            totals = {'alpha': sq(alpha), 'harmonic': sq(harmonic), 'coexact': sq(coexact), 'exact': sq(exact)}
            block = {}
            for name, arr in (('alpha', alpha), ('harmonic', harmonic), ('coexact', coexact)):
                block[f'{name}_norm2_total'] = totals[name]
                block.update({f'{name}_norm2_{part}': sq(arr[s]) for part, s in parts.items()})
            block['exact_norm2_total'] = totals['exact']
            block.update({f'{name}_fraction_total': _fraction(totals[name], totals['alpha'])
                          for name in ('harmonic', 'coexact', 'exact')})
            block.update({'beta_norm2': beta_norm2, 'beta_per_face_mean': beta_norm2 / max(1, n_f * k)})
            results.append({
                'n_layers': len(layer_subset), 'layer_indices': layer_subset, 'edge_dim_k': k,
                'topology': {**betti(node_idx, edges, faces), 'n_h': n_h, 'n_w': n_w, 'n_d': n_d},
                'n_h': n_h, 'n_v': n_w, 'n_d': n_d, 'n_f': n_f, 'block': block,
                'chain_residual': float(np.linalg.norm(d1 @ d0)) if n_f else 0.0,
                'reconstruction_error': float(np.linalg.norm(alpha - exact - harmonic - coexact)),
            })
        return {'model': self.model, 'n_pairs': self.n_pairs, 'edge_dim': self.edge_dim,
                'n_layers_total': n_layers_total, 'd_model': d_model, 'results': results}

    def _hypergraph(self):
        facts = build_facts(self.n_relations, self.n_per_fact)
        topos = {mode: build_fact_complex(facts, self.n_per_fact, mode, True)
                 for mode in ('matching', 'ring', 'triangulated')}
        order = list(topos['triangulated'].nodes)
        prompts = fact_prompts(facts, self.n_per_fact)
        model, tok, device = load_model(self.model, self.device)
        d_model = model.config.hidden_size
        feats = extract_layers(model, tok, [prompts[n] for n in order], [self.layer], device)[self.layer]
        del model
        torch.cuda.empty_cache()

        # Shared PCA across all node activations (identity transports).
        _, _, Vt = torch.linalg.svd(feats - feats.mean(dim=0, keepdim=True), full_matrices=False)
        k = min(self.edge_dim, Vt.shape[0])
        proj = feats @ Vt[:k].T.contiguous()
        row = {n: i for i, n in enumerate(order)}
        results = {name: self._hypergraph_topology(H, proj, row, k) for name, H in topos.items()}
        return {'model': self.model, 'layer': self.layer, 'edge_dim': self.edge_dim, 'actual_k': k,
                'n_relations': self.n_relations, 'n_per_fact': self.n_per_fact, 'n_probes': self.n_probes,
                'd_model': d_model, 'topologies': results}

    def _hypergraph_topology(self, H, proj, row, k):
        node_idx = {n: i for i, n in enumerate(H.nodes)}
        coh = betti(node_idx, H.edges, H.faces)
        d0, d1 = simplicial_coboundaries(node_idx, H.edges, H.faces)
        model_cochain = dict.fromkeys(('alpha_norm2', 'exact_fraction', 'harmonic_fraction', 'coexact_fraction',
                                       'beta_norm2', 'chain_residual', 'reconstruction_error'), 0.0)
        if H.edges:
            alpha = np.stack([(proj[row[b]] - proj[row[a]]).numpy() for a, b in H.edges]).astype(np.float64)
            exact, harm, coexact = hodge_2d(alpha, d0, d1)
            a2 = sq(alpha)
            model_cochain = {
                'alpha_norm2': a2, 'exact_fraction': _fraction(sq(exact), a2),
                'harmonic_fraction': _fraction(sq(harm), a2), 'coexact_fraction': _fraction(sq(coexact), a2),
                'beta_norm2': sq(d1 @ alpha) if d1.shape[0] else 0.0,
                'chain_residual': float(np.linalg.norm(d1 @ d0)) if d1.shape[0] else 0.0,
                'reconstruction_error': float(np.linalg.norm(alpha - exact - harm - coexact)),
            }
        return {
            'report': H.report(),
            'topology_alone': {'H0': coh['b_0'], 'H1': coh['b_1'], 'H2': coh['b_2']},
            'lifted_operator': {'n_v': coh['V'], 'n_e': coh['E'], 'n_f': coh['F'],
                                'rank_delta0_lifted': coh['rank_delta0'], 'rank_delta1_lifted': coh['rank_delta1'],
                                'H0_lifted': coh['b_0'], 'H1_lifted': coh['b_1'], 'H2_lifted': coh['b_2']},
            'harmonic_random_unit_cochain': hodge_random(d0, d1, k, self.n_probes) if H.edges else 0.0,
            'model_cochain': model_cochain,
        }


@dataclass
class HolonomyNull:
    """Null calibration for relation-hypergraph cycle holonomy.

    Small-N Procrustes holonomies approach the orthogonal bound sqrt(2k)
    regardless of structure, so the observed per-cycle ||I - H_gamma||_F is
    compared with a within-relation shuffled-paraphrase null (refit P_v and
    Q_ij per shuffle), a tokens-per-node bootstrap, and identity transports.
    """

    model: str = 'gpt2'
    layer: int = 8
    edge_dim: int = 8
    n_per_fact: int = 5
    n_token_samples: int = 6
    n_shuffle_seeds: int = 20
    bootstrap_n_values: Tuple[int, ...] = (2, 3, 4, 5, 6)
    quantize: Literal['none', '4bit'] = 'none'
    output_dir: str = 'outputs/holonomy_null'
    seed: int = 0

    def run(self):
        out_path = results_path(self.output_dir, self.model)
        # Per-fact paraphrase rings plus a per-relation ring through each fact's
        # first paraphrase: both within-fact and across-fact fundamental cycles.
        facts = build_facts(n_relations=6, paraphrases_per_fact=self.n_per_fact)
        H = build_fact_complex(facts, self.n_per_fact, 'ring', relation_rings=True)
        rel_to_facts = defaultdict(list)
        for i, fact in enumerate(facts):
            rel_to_facts[fact['relation_id']].append(i)
        cycles = find_fundamental_cycles(H.nodes, H.edges)

        model, tokenizer = load_causal_model(self.model, self.quantize, dtype=torch.bfloat16)
        device = next(model.parameters()).device
        features = {name: token_activations(model, tokenizer, prompt, self.layer, device, self.n_token_samples)
                    for name, prompt in fact_prompts(facts, self.n_per_fact).items()}
        del model
        torch.cuda.empty_cache()

        def obstructions(feats, seed, n_tokens=None):
            return holonomy_obstructions(feats, H.edges, cycles, self.edge_dim, device, seed, n_tokens)

        observed = obstructions(features, self.seed)
        shuffled = np.array([
            float(obstructions(shuffle_within_relation(features, rel_to_facts, self.n_per_fact, seed), seed).mean())
            for seed in range(self.seed + 1000, self.seed + 1000 + self.n_shuffle_seeds)
        ])
        bootstrap = {N: mean_std(obstructions({n: f[:N] for n, f in features.items()}, self.seed + N, N))
                     for N in self.bootstrap_n_values if N <= self.n_token_samples}
        eye = torch.eye(self.edge_dim, device=device)
        identity = cycle_obstructions(cycles, {e: eye.clone() for e in H.edges}, self.edge_dim, device).numpy()
        write_json(out_path, {
            'model': self.model, 'layer': self.layer, 'edge_dim': self.edge_dim, 'n_per_fact': self.n_per_fact,
            'n_token_samples': self.n_token_samples, 'n_nodes': len(H.nodes), 'n_edges': len(H.edges),
            'n_cycles': len(cycles), 'frobenius_upper_bound_sqrt_2k': float(np.sqrt(2 * self.edge_dim)),
            'observed': {**mean_std(observed), 'obstruction_per_cycle': observed.tolist()},
            'shuffled_null': {
                'n_shuffle_seeds': self.n_shuffle_seeds, 'obstruction_means_per_seed': shuffled.tolist(),
                'mean_of_means': float(shuffled.mean()), 'sd_of_means': float(shuffled.std()),
                'one_sided_p_shuffled_ge_observed': float(((shuffled >= observed.mean()).sum() + 1)
                                                          / (len(shuffled) + 1)),
            },
            'bootstrap_N': bootstrap,
            'identity_null': mean_std(identity),
            'seed': self.seed,
        })


@dataclass
class Scalability:
    """Timing of the sheaf construction across hidden sizes (no model load).

    The eigenproblem lives in the k x k edge space whatever the hidden
    dimension; this times fit_h0_pca (joint-PCA fit + edge-space
    eigendecomposition) and the k x k eigendecomposition alone on random
    activations for d from GPT-2 (768) to Llama-2-13B (5120).
    """

    dims: Tuple[int, ...] = (768, 2048, 2560, 3584, 4096, 5120)
    n_pairs: int = 200
    edge_dim: int = 128
    n_repeat: int = 20
    output_dir: str = 'outputs/scalability'
    seed: int = 0

    def run(self):
        rng, k, rows = np.random.default_rng(self.seed), self.edge_dim, []
        for d in self.dims:
            a = torch.tensor(rng.standard_normal((self.n_pairs, d)).astype(np.float32))
            b = a + 0.1 * torch.tensor(rng.standard_normal((self.n_pairs, d)).astype(np.float32))
            full_med, full_std = self._time(lambda: fit_h0_pca(a, b, k=20, edge_dim=k))
            Dk = (a - b).numpy() @ joint_pca_projection(a, b, k)
            L = (Dk.T @ Dk) / self.n_pairs
            eig_med, eig_std = self._time(lambda: np.linalg.eigh(L))
            rows.append({'d': d, 'full_ms': full_med * 1000, 'full_std_ms': full_std * 1000,
                         'eig_ms': eig_med * 1000, 'eig_std_ms': eig_std * 1000})
        write_json(results_path(self.output_dir, 'scalability'), {
            'n_pairs': self.n_pairs, 'k': k, 'n_repeat': self.n_repeat, 'rows': rows,
            'eig_span': max(r['eig_ms'] for r in rows) / min(r['eig_ms'] for r in rows),
            'full_span': max(r['full_ms'] for r in rows) / min(r['full_ms'] for r in rows),
        })

    def _time(self, fn):
        """Median and std wall-clock seconds over ``n_repeat`` calls after two warmups."""
        fn()
        fn()
        times = []
        for _ in range(self.n_repeat):
            t0 = time.perf_counter()
            fn()
            times.append(time.perf_counter() - t0)
        return float(np.median(times)), float(np.std(times))


EXPERIMENTS = {
    'grid': Grid,
    'hypergraph': Hypergraph,
    'triangulated': Triangulated,
    'holonomy_null': HolonomyNull,
    'scalability': Scalability,
}

if __name__ == '__main__':
    tyro.extras.subcommand_cli_from_dict(EXPERIMENTS, description=__doc__, use_underscores=True).run()
