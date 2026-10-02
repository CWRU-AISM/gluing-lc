"""
Held-out fact retrieval: sheaf H^0 against LEACE and other subspaces.

    python experiments/retrieval.py <experiment> --model <hf id> [flags]

Each experiment fits its bases on training facts and scores top-1 retrieval of
held-out paraphrases against per-fact centroids (hard = same-relation
candidates only). Results go to outputs/<dir>/<model>_<timestamp>.json.
"""

from dataclasses import dataclass
from functools import partial
from typing import Literal, Optional, Tuple

import numpy as np
import torch
import tyro

from sheafint import (
    clique_edges,
    count_clique_cycles,
    count_ring_cycles,
    fit_h0_from_edges,
    fit_h0_identity,
    fit_h0_pca,
    fit_h0_random,
    pair_edges,
    ring_edges,
)
from sheafint.core.h0 import joint_pca_projection
from sheafint.data import build_facts
from sheafint.data.relation_templates import RELATION_TEMPLATES
from utils.baselines import (
    cca_basis, embed_sentence_transformer, embed_simcse, lda_basis, nca_basis, pca_top_basis, pls_basis,
    whitened_pca_map,
)
from utils.cycle_arms import MIN_EXPR, bootstrap_contrasts, build_arms, nname
from utils.datasets import load_paraphrase_pairs, load_translation_pairs
from utils.hodge import load_model
from utils.leace import (
    centroid_lookup, fit_leace_eraser, fit_pair_eraser, paraphrase_pairs, random_first_paraphrases, top_pcs,
)
from utils.model_io import load_causal_model, n_layers, pooled_hidden_states
from utils.results import results_path, write_json
from utils.retrieval import (
    HeldOutConfig, HeldOutData, HeldOutRetrieval, cosine_sim, encode, fit_random_bases, fit_sheaf_h1_basis,
    make_train_pairs,
)

FP32_POOL = dict(max_length=64, fp32_pool=True)


@dataclass
class Leace(HeldOutConfig):
    """Table 2: sheaf H^0 vs LEACE on held-out CounterFact.

    H^0 (joint-PCA P, bottom-k eigenvectors of L_F) and a LEACE eraser are fit
    on training facts. LEACE_preserved scores the full (d-1)-dim preserved
    subspace and LEACE_preserved_{k}d its top-k PCs (rank-matched to H^0);
    PCA_top{k} and Random_{k}d are the unsupervised and chance baselines.
    """

    output_dir: str = 'outputs/leace_counterfact'
    n_random_trials: int = 5

    def run(self):
        out_path = results_path(self.output_dir, self.model)
        data = HeldOutRetrieval.from_args(self)
        (H_a, H_b, cent_t, qry_t), layer, d = encode(
            self, [*make_train_pairs(data.facts, data.train_ids), data.centroid_prompts, data.query_prompts])
        k = self.k
        cent, qry = cent_t.numpy(), qry_t.numpy()
        eraser = fit_leace_eraser(H_a, self.seed)
        cent_l, qry_l = eraser(cent_t).numpy(), eraser(qry_t).numpy()
        leace_k = top_pcs(eraser(H_a).numpy(), k)
        bases = {'Full': None, f'H0_sheaf_{k}d': fit_h0_pca(H_a, H_b, k=k, edge_dim=self.edge_dim),
                 f'PCA_top{k}': joint_pca_projection(H_a, H_b, k)}
        rand = fit_random_bases(d, k, self.n_random_trials, self.seed)

        lookups = {}
        for name, hard in (('easy_lookup_held_out', False), ('hard_lookup_held_out', True)):
            m = {b: data.score(qry, cent, basis, hard) for b, basis in bases.items()}
            m['LEACE_preserved'] = data.score(qry_l, cent_l, None, hard)
            m[f'LEACE_preserved_{k}d'] = data.score(qry_l, cent_l, leace_k, hard)
            m[f'Random_{k}d'] = float(np.mean([data.score(qry, cent, b, hard) for b in rand]))
            lookups[name] = m
        write_json(out_path, {**data.run_info(self, layer, d), 'k': k, 'n_train_pairs': int(H_a.shape[0]),
                              'n_random_trials': self.n_random_trials, **lookups})


@dataclass
class LeaceTemplated:
    """Appendix LEACE table: sheaf H^0 vs LEACE on the 60 templated facts.

    All paraphrases of sheafint.data.build_facts() are embedded in bf16 at
    the depth set by --layer_frac. H^0 is fit on consecutive paraphrase pairs;
    LEACE erases paraphrase-pair member vs first paraphrase of a random fact.
    The table reports easy_lookup (top-1 among all facts); hard_lookup
    restricts candidates to the query's relation.
    """

    model: str
    """Hugging Face model id."""
    n_facts: int = 200
    """Capped at the 60 templated facts."""
    paraphrases_per_fact: int = 5
    layer_frac: float = 0.67
    k: int = 20
    edge_dim: int = 128
    n_train_pairs: int = 200
    output_dir: str = 'outputs/leace_multi'
    device: Optional[str] = None

    def run(self):
        out_path = results_path(self.output_dir, self.model)
        facts = build_facts(n_relations=6, paraphrases_per_fact=self.paraphrases_per_fact)[:self.n_facts]
        n_par, n_facts = self.paraphrases_per_fact, len(facts)
        labels = np.repeat(np.arange(n_facts), n_par)
        relations = np.repeat([list(RELATION_TEMPLATES).index(f['relation_id']) for f in facts], n_par)

        model, tokenizer, _ = load_model(self.model, self.device)
        depth, d = n_layers(model), model.config.hidden_size
        layer = int(self.layer_frac * depth)
        H_t = pooled_hidden_states(model, tokenizer, [p for f in facts for p in f['paraphrases'][:n_par]],
                                   layer, batch_size=8, **FP32_POOL)
        del model
        torch.cuda.empty_cache()

        H = H_t.numpy()
        pos_a, pos_b = paraphrase_pairs(H, n_facts, n_par)
        negatives = random_first_paraphrases(H, n_facts, n_par, min(self.n_train_pairs, len(pos_a)))
        eraser = fit_pair_eraser(pos_a, negatives)
        preserved = eraser(H_t).numpy()
        k = self.k
        rand, _ = np.linalg.qr(np.random.default_rng(42).standard_normal((d, k)).astype(np.float32))
        bases = {'Full': None, f'H0_sheaf_{k}d': fit_h0_pca(pos_a, pos_b, k=k, edge_dim=self.edge_dim),
                 f'PCA_top{k}': np.linalg.svd(H - H.mean(axis=0, keepdims=True), full_matrices=False)[2][:k].T,
                 f'Random_{k}d': rand}

        def lookup(mask):
            out = {name: centroid_lookup(H[mask], labels[mask], b) for name, b in bases.items()}
            out['LEACE_preserved'] = centroid_lookup(preserved[mask], labels[mask])
            out['LEACE_erased'] = centroid_lookup((H - preserved)[mask], labels[mask])
            return out

        per_relation = [lookup(relations == r) for r in np.unique(relations) if (relations == r).sum() >= 2 * n_par]
        write_json(out_path, {
            'model': self.model, 'n_layers': depth, 'hidden_dim': d, 'layer': layer, 'n_facts': n_facts,
            'paraphrases_per_fact': n_par, 'easy_lookup': lookup(np.ones(len(H), dtype=bool)),
            'hard_lookup': {name: float(np.mean([r[name] for r in per_relation])) for name in per_relation[0]},
        })


@dataclass
class Restriction(HeldOutConfig):
    """Restriction-map ablation: joint-PCA P vs random orthonormal P vs no projection.

    Tests whether the kernel selection or the joint-PCA map carries the H^0
    retrieval signal (hard same-relation lookup).
    """

    output_dir: str = 'outputs/restriction_ablation'
    n_random_trials: int = 5

    def run(self):
        out_path = results_path(self.output_dir, self.model)
        data = HeldOutRetrieval.from_args(self)
        (H_a, H_b, cent, qry), layer, d = encode(
            self, [*make_train_pairs(data.facts, data.train_ids), data.centroid_prompts, data.query_prompts])
        cent, qry, k = cent.numpy(), qry.numpy(), self.k
        bases = {'Full': None, 'H0_PCA_P': fit_h0_pca(H_a, H_b, k=k, edge_dim=self.edge_dim),
                 'H0_identity_P': fit_h0_identity(H_a, H_b, k=k), f'PCA_{k}': joint_pca_projection(H_a, H_b, k)}
        hard = {name: data.score(qry, cent, b) for name, b in bases.items()}
        h0_random = [data.score(qry, cent, fit_h0_random(H_a, H_b, k=k, edge_dim=self.edge_dim,
                                                          seed=self.seed + 100 + s))
                     for s in range(self.n_random_trials)]
        rand = [data.score(qry, cent, b) for b in fit_random_bases(d, k, self.n_random_trials, self.seed)]
        hard.update({'H0_random_P_mean': float(np.mean(h0_random)), 'H0_random_P_std': float(np.std(h0_random)),
                     f'Random_{k}_mean': float(np.mean(rand)), f'Random_{k}_std': float(np.std(rand))})
        write_json(out_path, {**data.run_info(self, layer, d), 'k': k, 'n_random_trials': self.n_random_trials,
                              'hard_lookup_held_out': hard})


@dataclass
class Cycle(HeldOutConfig):
    """Cycle-aware H^0: pair-only forest vs multi-paraphrase rings and cliques.

    The pair graph is acyclic (b_1 = 0); rings (b_1 = 1 per fact) and cliques
    (b_1 = (K-1)(K-2)/2) require invariance across all paraphrases of a fact.
    """

    output_dir: str = 'outputs/cycle_h0'

    def run(self):
        out_path = results_path(self.output_dir, self.model)
        data = HeldOutRetrieval.from_args(self)
        facts, train_ids = data.facts, data.train_ids
        (pair_e, pair_n), (ring_e, ring_n), (clique_e, clique_n) = (
            build(facts, train_ids) for build in (pair_edges, ring_edges, clique_edges))
        nodes = sorted(set(pair_n + ring_n + clique_n))
        prompt_of = {f'f{i}_p{j}': e for i in train_ids for j, e in enumerate(facts[i]['expressions'])}
        (H, cent, qry), layer, d = encode(self, [[prompt_of[n] for n in nodes], data.centroid_prompts,
                                                 data.query_prompts])
        acts = dict(zip(nodes, H.numpy()))
        pair_acts = torch.from_numpy(np.stack([acts[n] for n in pair_n]))
        bases = {'Full': None, **{f'H0_{name}': fit_h0_from_edges(acts, e, k=self.k, edge_dim=self.edge_dim)
                                  for name, e in (('pair_only', pair_e), ('multi_paraphrase_ring', ring_e),
                                                  ('multi_paraphrase_clique', clique_e))},
                 f'PCA_top{self.k}': joint_pca_projection(pair_acts, pair_acts, self.k)}
        write_json(out_path, {
            **data.run_info(self, layer, d), 'k': self.k,
            'n_pair_edges': len(pair_e), 'n_ring_edges': len(ring_e), 'n_clique_edges': len(clique_e),
            'b1_ring': count_ring_cycles(facts, train_ids), 'b1_clique': count_clique_cycles(facts, train_ids),
            'hard_lookup_held_out': {name: data.score(qry.numpy(), cent.numpy(), b) for name, b in bases.items()},
        })


CYCLE_VARIANTS = {
    'matched': dict(n_facts=600, output_dir='outputs/cycle_h0_matched',
                    contrasts=[('ring_minus_acyclic_matched', 'ring', 'acyclic_matched'),
                               ('ring_minus_spanning_tree', 'ring', 'spanning_tree')]),
    'grid': dict(n_facts=400, output_dir='outputs/cycle_h0_matched_grid',
                 contrasts=[('grid_minus_tree6', 'grid', 'grid_tree6')]),
    'hyper': dict(n_facts=600, output_dir='outputs/cycle_h0_matched_hyper',
                  contrasts=[('hypergraph_minus_acyclic', 'hypergraph', 'hyper_acyclic')]),
}


@dataclass
class CycleMatched(HeldOutConfig):
    """Edge- and node-matched acyclic controls for cycle-aware H^0.

    matched: pair_only / spanning_tree (node-matched) / acyclic_matched
    (edge-matched) / ring / clique. grid: (paraphrase, layer) nodes over 0.5 L
    and the eval layer, b_1 = 4 per fact, vs its spanning trees. hyper:
    per-fact rings plus per-relation rings vs node- and edge-matched paths.
    Paired-bootstrap CIs on the cyclic minus acyclic contrasts. All arms use
    facts with >= 6 paraphrases; --n_facts defaults to 600 (400 for grid).
    """

    n_facts: Optional[int] = None
    quantize: Literal['none', '4bit'] = 'none'
    output_dir: Optional[str] = None
    variant: Literal['matched', 'grid', 'hyper'] = 'matched'
    n_boot: int = 300

    def run(self):
        spec = CYCLE_VARIANTS[self.variant]
        self.n_facts = self.n_facts or spec['n_facts']
        out_path = results_path(self.output_dir or spec['output_dir'], self.model)
        data = HeldOutRetrieval.from_args(self, min_expressions=MIN_EXPR)
        facts, train_ids = data.facts, data.train_ids
        arms, extra = build_arms(self.variant, facts, train_ids)
        all_nodes = sorted(set().union(*[set(n) for _, n in arms.values()]))
        prompt_of = {(i, j): ex for i in train_ids for j, ex in enumerate(facts[i]['expressions'])}
        used = [(int(n.split('_')[0][1:]), int(n.split('_')[1][1:])) for n in all_nodes]
        if self.variant == 'grid':
            used = sorted(set(used))  # both layers share one prompt list

        model, tokenizer = load_causal_model(self.model, quantize=self.quantize)
        depth = n_layers(model)
        eval_layer = int(self.layer_frac * depth)
        layers = {0: int(0.5 * depth), 1: eval_layer} if self.variant == 'grid' else {None: eval_layer}

        pool = partial(pooled_hidden_states, model, tokenizer, batch_size=self.batch_size, **FP32_POOL)
        acts = {nname(i, j, l): vec for l, lidx in layers.items()
                for (i, j), vec in zip(used, pool([prompt_of[u] for u in used], lidx).numpy())}
        cent, qry = (pool(prompts, eval_layer).numpy() for prompts in (data.centroid_prompts, data.query_prompts))
        del model, pool
        torch.cuda.empty_cache()

        bases = {name: fit_h0_from_edges(acts, e, k=self.k, edge_dim=self.edge_dim) for name, (e, _) in arms.items()}
        if self.variant == 'matched':
            H = torch.from_numpy(np.stack([acts[n] for n in all_nodes]))
            bases[f'PCA_top{self.k}'] = joint_pca_projection(H, H, self.k)
        hard = {'Full': data.score(qry, cent)} if self.variant == 'matched' else {}
        hard.update({name: data.score(qry, cent, b) for name, b in bases.items()})
        cis = bootstrap_contrasts(lambda b, rows: data.score(qry, cent, b, rows=rows), bases, spec['contrasts'],
                                  len(qry), self.n_boot, np.random.default_rng(self.seed))

        payload = {'model': self.model, ('layers' if self.variant == 'grid' else 'layer'):
                   layers if self.variant == 'grid' else eval_layer, 'seed': self.seed,
                   'n_train_facts': len(train_ids)}
        if self.variant == 'matched':
            payload['n_test_facts'] = len(data.test_ids)
        payload.update({'n_hard_queries': len(qry), **extra,
                        'edges_per_arm': {name: len(e) for name, (e, _) in arms.items()},
                        'hard_retrieval': {name: float(v) for name, v in hard.items()}})
        payload.update({f'{key}_pp': float((hard[a] - hard[b]) * 100) for key, a, b in spec['contrasts']})
        payload.update({f'{key}_ci95': ci for key, ci in cis.items()})
        write_json(out_path, payload)


@dataclass
class Crossdataset(HeldOutConfig):
    """Cross-dataset restriction-map transfer.

    H^0 is fit on out-of-domain paraphrase pairs (MRPC, PAWS, QQP) and scored on
    held-out hard CounterFact, against in-domain CounterFact H^0, PCA-k and
    Random-k.
    """

    output_dir: str = 'outputs/crossdataset_p'
    n_paraphrase_pairs: int = 300
    datasets: Tuple[str, ...] = ('mrpc', 'paws', 'qqp')

    def run(self):
        out_path = results_path(self.output_dir, self.model)
        data = HeldOutRetrieval.from_args(self)
        lists = [*make_train_pairs(data.facts, data.train_ids)]
        for name in self.datasets:
            pairs = load_paraphrase_pairs(name, self.n_paraphrase_pairs, seed=self.seed)
            lists += [[p[0] for p in pairs], [p[1] for p in pairs]]
        acts, layer, d = encode(self, lists + [data.centroid_prompts, data.query_prompts])
        (cf_a, cf_b), cent, qry = acts[:2], acts[-2].numpy(), acts[-1].numpy()
        k = self.k

        bases = {'Full': None, 'H0_counterfact': fit_h0_pca(cf_a, cf_b, k=k, edge_dim=self.edge_dim)}
        for i, name in enumerate(self.datasets):
            bases[f'H0_{name}'] = fit_h0_pca(acts[2 + 2 * i], acts[3 + 2 * i], k=k, edge_dim=self.edge_dim)
        bases[f'PCA_top{k}'] = joint_pca_projection(cf_a, cf_b, k)
        hard = {name: data.score(qry, cent, b) for name, b in bases.items()}
        rand = [data.score(qry, cent, b) for b in fit_random_bases(d, k, 5, self.seed)]
        hard.update({f'Random_{k}_mean': float(np.mean(rand)), f'Random_{k}_std': float(np.std(rand))})
        write_json(out_path, {**data.run_info(self, layer, d), 'k': k,
                              'n_paraphrase_pairs_per_source': self.n_paraphrase_pairs,
                              'datasets_loaded': list(self.datasets), 'hard_lookup_held_out': hard})


@dataclass
class MSweep(HeldOutData):
    """H^0 dimension m sweep; m = edge_dim is the no-kernel-selection control."""

    output_dir: str = 'outputs/m_sensitivity'
    m_values: Tuple[int, ...] = (5, 10, 20, 40, 80, 128)

    def run(self):
        out_path = results_path(self.output_dir, self.model)
        data = HeldOutRetrieval.from_args(self)
        (H_a, H_b, cent, qry), layer, d = encode(
            self, [*make_train_pairs(data.facts, data.train_ids), data.centroid_prompts, data.query_prompts])
        cent, qry = cent.numpy(), qry.numpy()
        bases = {str(m): fit_h0_pca(H_a, H_b, k=min(m, self.edge_dim), edge_dim=self.edge_dim)
                 for m in self.m_values}
        write_json(out_path, {
            **data.run_info(self, layer, d), 'm_values': list(self.m_values),
            'hard_lookup_by_m': {m: data.score(qry, cent, b, hard=True) for m, b in bases.items()},
            'easy_lookup_by_m': {m: data.score(qry, cent, b, hard=False) for m, b in bases.items()},
            'hard_lookup_full_dim': data.score(qry, cent, None, hard=True),
            'easy_lookup_full_dim': data.score(qry, cent, None, hard=False),
        })


@dataclass
class Pooling(HeldOutConfig):
    """Pooling sensitivity: mean over non-pad tokens, last non-pad token, first token."""

    output_dir: str = 'outputs/pooling_sensitivity'
    poolings: Tuple[Literal['mean', 'last', 'first'], ...] = ('mean', 'last', 'first')

    def run(self):
        out_path = results_path(self.output_dir, self.model)
        data = HeldOutRetrieval.from_args(self)
        texts = [*make_train_pairs(data.facts, data.train_ids), data.centroid_prompts, data.query_prompts]
        model, tokenizer = load_causal_model(self.model, quantize=self.quantize)
        layer, d = int(self.layer_frac * n_layers(model)), model.config.hidden_size

        results = {}
        for pooling in self.poolings:
            H_a, H_b, cent, qry = [pooled_hidden_states(model, tokenizer, t, layer, pooling=pooling,
                                                        batch_size=self.batch_size, **FP32_POOL) for t in texts]
            cent, qry = cent.numpy(), qry.numpy()
            h0 = fit_h0_pca(H_a, H_b, k=self.k, edge_dim=self.edge_dim)
            results[pooling] = {'Full_hard': data.score(qry, cent, None, hard=True),
                                'H0_sheaf_hard': data.score(qry, cent, h0, hard=True),
                                'Full_easy': data.score(qry, cent, None, hard=False),
                                'H0_sheaf_easy': data.score(qry, cent, h0, hard=False)}
        write_json(out_path, {**data.run_info(self, layer, d), 'k': self.k, 'poolings': list(self.poolings),
                              'results_by_pooling': results})


@dataclass
class Baselines(HeldOutConfig):
    """Retrieval baselines with the leace protocol (same facts, split, scorer).

    Activation subspaces of the same frozen activations: LDA, CCA, PLS, top-k
    PCA, whitened PCA (1/sqrt(eigenvalue) on centred activations), NCA.
    SimCSE and SBERT embed the raw texts and are reference points only.
    """

    quantize: Literal['none', '4bit'] = 'none'
    output_dir: str = 'outputs/retrieval_baselines'

    def run(self):
        out_path = results_path(self.output_dir, self.model)
        data = HeldOutRetrieval.from_args(self)
        train_fact_id = np.array([i for i in data.train_ids
                                  for _ in range(min(3, len(data.facts[i]['expressions']) - 1))])
        acts, layer, d = encode(self, [*make_train_pairs(data.facts, data.train_ids), data.centroid_prompts,
                                       data.query_prompts], **FP32_POOL)
        A, B, cent, qry = (a.numpy() for a in acts)
        X, labels, k = np.concatenate([A, B], axis=0), np.concatenate([train_fact_id, train_fact_id]), self.k
        bases = {
            'H0_sheaf': fit_h0_pca(torch.from_numpy(A), torch.from_numpy(B), k=k, edge_dim=self.edge_dim),
            'LDA': lda_basis(X, labels, k, self.edge_dim), 'CCA': cca_basis(A, B, k, self.edge_dim),
            'PLS': pls_basis(A, B, k, self.edge_dim), 'PCA_top': pca_top_basis(X, k),
            'NCA_metric': nca_basis(X, labels, k, self.edge_dim, self.seed), 'Full': None,
        }

        def both(q, c, basis=None):
            return {'easy': data.score(q, c, basis, hard=False), 'hard': data.score(q, c, basis, hard=True)}

        results = {name: both(qry, cent, b) for name, b in bases.items()}
        W, mu = whitened_pca_map(X, k)
        results['whitened_PCA'] = both(qry - mu, cent - mu, W)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        for name, embed in (('SimCSE_sup', embed_simcse), ('SBERT_mpnet', embed_sentence_transformer)):
            results[name] = {**both(embed(data.query_prompts, device), embed(data.centroid_prompts, device)),
                             'note': 'external encoder on raw text'}
        write_json(out_path, {**data.run_info(self, layer, d), 'k': k, 'n_facts': self.n_facts,
                              'train_frac': self.train_frac, 'results': results})


@dataclass
class Crosslingual:
    """Cross-lingual probe: EN<->FR translation pairs (OPUS-100) as sheaf edges.

    H^0 / H^1 are fit on the first half of the pairs and scored on the disjoint
    second half (fitting and scoring on the same pairs saturates H^0).
    """

    model: str = 'meta-llama/Meta-Llama-3-8B'
    n_pairs: int = 200
    layer_frac: float = 0.67
    k: int = 20
    edge_dim: int = 128
    quantize: Literal['none', '4bit'] = '4bit'
    output_dir: str = 'outputs/crosslingual_probe'
    seed: int = 42
    batch_size: int = 8

    def run(self):
        out_path = results_path(self.output_dir, self.model)
        en, fr = load_translation_pairs(self.n_pairs, self.seed)
        (A, B), layer, _ = encode(self, [en, fr], **FP32_POOL)
        A, B, half = A.numpy(), B.numpy(), len(en) // 2
        A_tr, B_tr, A_te, B_te = torch.tensor(A[:half]), torch.tensor(B[:half]), A[half:], B[half:]
        bases = {'Full': None, 'H0': fit_h0_pca(A_tr, B_tr, k=self.k, edge_dim=self.edge_dim),
                 'H1': fit_sheaf_h1_basis(A_tr, B_tr, self.k, self.edge_dim)[0]}

        def project(X, basis):
            return X @ basis if basis is not None else X

        def aligned_top1(query, pool, basis):
            preds = np.argmax(cosine_sim(project(query, basis), project(pool, basis)), axis=1)
            return float((preds == np.arange(len(query))).mean())

        def within_pair_cosine(basis):
            X, Y = project(A_te, basis), project(B_te, basis)
            X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-10)
            Y = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-10)
            return float((X * Y).sum(axis=1).mean())

        results = {name: {'within_pair_cosine': within_pair_cosine(b),
                          'retrieval_fr_to_en': aligned_top1(B_te, A_te, b),
                          'retrieval_en_to_fr': aligned_top1(A_te, B_te, b)} for name, b in bases.items()}
        write_json(out_path, {'model': self.model, 'layer': layer, 'n_pairs': len(en), 'n_train': half,
                              'n_test': len(en) - half, 'k': self.k, 'edge_dim': self.edge_dim, 'seed': self.seed,
                              'results': results})


EXPERIMENTS = {
    'leace': Leace,
    'leace_templated': LeaceTemplated,
    'restriction': Restriction,
    'cycle': Cycle,
    'cycle_matched': CycleMatched,
    'crossdataset': Crossdataset,
    'm_sweep': MSweep,
    'pooling': Pooling,
    'baselines': Baselines,
    'crosslingual': Crosslingual,
}

if __name__ == '__main__':
    tyro.extras.subcommand_cli_from_dict(EXPERIMENTS, description=__doc__, use_underscores=True).run()
