"""
CounterFact steering: sheaf steering vectors, the generation loop, and scoring.

Restriction maps (joint PCA, and CCA / contrastive refining it) feed the
sheaf-Laplacian and Fisher decompositions; ``steering_vectors`` turns each
into H^1-dims / H^0-removed vectors next to the shared baselines. Each
steered generation is scored by :func:`text_metrics.evaluate_generation`
(0/1 fact, subject, style, coherence, joint), and per-prompt scores become
bootstrap summaries and paired McNemar / Mann-Whitney comparisons.
"""

from typing import Callable, Dict, List, Optional

import numpy as np
import torch
from scipy.stats import mannwhitneyu
from tqdm import tqdm

from sheafint.steering import (
    baseline_vectors, cca, contrastive, decomposition_vectors, fisher_decomposition, joint_pca,
    sheaf_laplacian_decomposition,
)

from .model_io import generate
from .statistics import bootstrap_ci, mcnemar_test
from .text_metrics import evaluate_generation

METRICS = ('fact', 'subject', 'style', 'coherent', 'joint', 'coherent_joint')


@torch.no_grad()
def median_token_norm(model, tokenizer, texts, layer: int, max_length: int = 128) -> float:
    """Median hidden-state norm at ``layer`` over the non-pad tokens of ``texts`` (one batch).

    Robust to attention-sink tokens whose outlier norm inflates the
    mean-pooled scale 1.5-18x on some models.
    """
    device = next(model.parameters()).device
    enc = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=max_length)
    enc = {k: v.to(device) for k, v in enc.items()}
    hs = model(**enc, output_hidden_states=True).hidden_states[layer]
    return float(hs.norm(dim=-1)[enc['attention_mask'].bool()].median())


def restriction_maps(s1, s2, background, edge_dim, names):
    """``{name: P}`` for the requested maps plus per-map metadata; CCA / contrastive refine joint PCA."""
    P_pca, explained = joint_pca(s1, s2, edge_dim)
    maps, meta = {}, {}
    if 'pca' in names:
        maps['pca'], meta['pca'] = P_pca, {'var_explained': explained}
    if 'cca' in names:
        maps['cca'], corr = cca(s1, s2, P_pca)
        meta['cca'] = {'min_correlation': float(corr[0]), 'max_correlation': float(corr[-1]),
                       'median_correlation': float(np.median(corr))}
    if 'contrastive' in names:
        maps['contrastive'], explained_c, eigvals = contrastive(s1, s2, background, P_pca)
        meta['contrastive'] = {'var_explained': explained_c,
                               'eigval_range': [float(eigvals[-1]), float(eigvals[0])]}
    return maps, meta


def decompositions(s1, s2, background, maps, k):
    """``{'{map}_sheaf': ..., '{map}_fisher': ...}`` for every restriction map."""
    out = {}
    for name, P in maps.items():
        out[f'{name}_sheaf'] = sheaf_laplacian_decomposition(s1, s2, background, P, k=k)
        out[f'{name}_fisher'] = fisher_decomposition(s1, s2, background, P, k=k)
    return out


def steering_vectors(style, decomps, rng):
    """Shared baselines plus ``{group}_h1_dims`` / ``{group}_h0_removed`` per decomposition."""
    first = next(iter(decomps.values()))
    methods = baseline_vectors(style, first['var_h1'], rng, mod_vector=style)
    for group, decomp in decomps.items():
        methods.update({f'{group}_{m}': v for m, v in decomposition_vectors(style, decomp).items()})
    return methods


def decomposition_meta(decomp):
    return {
        'method': decomp['method'],
        'eigval_range': [float(decomp['eigvals'][0]), float(decomp['eigvals'][-1])],
        'h0_dim': int(decomp.get('h0_dim', -1)),
        'spectral_gap': float(decomp.get('spectral_gap', 0)),
        'h1_dims': decomp['h1_dims'],
        'var_h1': decomp['var_h1'],
        'h1_var_overlap': len(set(decomp['h1_dims']) & set(decomp['var_h1'])),
    }


def evaluate(model, tokenizer, test_data, generators: Dict[str, Callable[[str], str]], max_new_tokens: int,
             desc: str = 'Evaluating'):
    """Score every steered generator on every CounterFact prompt against the unsteered generation.

    A prompt whose generation fails scores ``None`` for every method.
    """
    results = {name: [] for name in generators}
    texts = []
    for idx, entry in enumerate(tqdm(test_data, desc=desc)):
        prompt = entry['gen_prompt']
        try:
            baseline = generate(model, tokenizer, prompt, max_new_tokens)
            row = {'idx': idx, **{k: entry[k] for k in ('case_id', 'gen_prompt', 'entity', 'subject', 'fact')},
                   'baseline': baseline[:500]}
            scores = {}
            for name, steer in generators.items():
                steered = steer(prompt)
                scores[name] = evaluate_generation(baseline, steered, entry['entity'], entry['subject'])
                row[f'{name}_output'] = steered[:500]
                row[f'{name}_eval'] = scores[name]
        except Exception:
            scores, row = dict.fromkeys(generators), None
        for name, score in scores.items():
            results[name].append(score)
        if row is not None:
            texts.append(row)
    return results, texts


def _pct(values, seed: int) -> Dict[str, float]:
    mean, lo, hi = bootstrap_ci(values, seed=seed)
    return {'mean': round(mean * 100, 1), 'ci_lo': round(lo * 100, 1), 'ci_hi': round(hi * 100, 1)}


def summarize_method(results: List[Optional[dict]], seed: int) -> Dict:
    """Bootstrap percentage and CI per metric over the prompts that generated.

    ``fact_given_known`` restricts fact preservation to prompts whose
    unsteered generation already contained the target entity.
    """
    valid = [r for r in results if r is not None]
    if not valid:
        return {'n': 0}
    metrics = {m: {**_pct([r[m] for r in valid], seed), 'raw': [r[m] for r in valid]} for m in METRICS}
    known = [r['fact'] for r in valid if r.get('known')]
    if known:
        metrics['fact_given_known'] = {**_pct(known, seed), 'n_known': len(known)}
    return {'n': len(valid), 'metrics': metrics}


def compare(s1: List[int], s2: List[int]) -> Dict[str, float]:
    """Paired McNemar and Mann-Whitney tests of two 0/1 outcome lists."""
    n = min(len(s1), len(s2))
    s1, s2 = s1[:n], s2[:n]
    try:
        _, p_mw = mannwhitneyu(s1, s2, alternative='two-sided')
    except ValueError:
        p_mw = 1.0
    return {
        'effect': round(float(np.mean(s1) - np.mean(s2)), 4),
        'ratio': round(float(np.mean(s1) / (np.mean(s2) + 1e-10)), 2),
        'p_mcnemar': round(mcnemar_test(s1, s2), 6),
        'p_mannwhitney': round(float(p_mw), 6),
    }


def contrasts(summary: Dict, pairs, metrics) -> Dict[str, Dict]:
    """``compare`` for every (m1, m2) in ``pairs`` and metric, skipping methods that did not run."""
    out = {}
    for metric in metrics:
        for m1, m2 in pairs:
            if summary.get(m1, {}).get('n', 0) and summary.get(m2, {}).get('n', 0):
                out[f'{m1}_vs_{m2}_{metric}'] = compare(summary[m1]['metrics'][metric]['raw'],
                                                        summary[m2]['metrics'][metric]['raw'])
    return out


def strip_raw(summary: Dict) -> Dict:
    """Drop per-prompt values before serialization."""
    return {
        name: {'n': s['n'], 'metrics': {m: {k: v for k, v in vals.items() if k != 'raw'}
                                        for m, vals in s['metrics'].items()}} if s['n'] else {'n': 0}
        for name, s in summary.items()
    }
