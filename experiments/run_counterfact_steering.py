#!/usr/bin/env python3
"""
Sheaf cohomology steering evaluation on CounterFact.

Builds joint-PCA restriction maps, sheaf-Laplacian and Fisher
decompositions, and evaluates random / full-style / variance-matched /
H1-dim / H0-removed steering with the McNemar paired test (Table 4).
"""

import argparse
import gc
import json
import sys
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
from scipy.stats import mannwhitneyu
from tqdm import tqdm
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sheafint.steering import (
    fisher_decomposition,
    joint_pca,
    sheaf_laplacian_decomposition,
)
from sheafint.steering.methods import build_steering_vectors, scale_to_norm
from experiments.utils.datasets import load_counterfact_data, load_mrpc_pairs
from experiments.utils.model_io import (
    generate as model_generate,
    generate_with_steering,
    load_causal_model,
    n_layers as model_n_layers,
    pooled_hidden_states,
)
from experiments.utils.statistics import bootstrap_ci, mcnemar_test
from experiments.utils.style_examples import CASUAL_EXAMPLES, FORMAL_EXAMPLES
from experiments.utils.text_metrics import evaluate_generation


def _build_decompositions(s_left, s_right, background, edge_dim, k):
    # Build both sheaf-Laplacian and Fisher decompositions over a shared PCA projection.
    projection, _ = joint_pca(s_left, s_right, edge_dim)
    return {
        'sheaf': sheaf_laplacian_decomposition(s_left, s_right, background, projection, k=k),
        'fisher': fisher_decomposition(s_left, s_right, background, projection, k=k),
    }


def _assemble_steering_vectors(decompositions, style_np, rng):
    # Combine shared baselines (random / variance / full / mod) with per-decomposition vectors.
    base_decomp = next(iter(decompositions.values()))
    shared = build_steering_vectors(style_np, base_decomp, rng, mod_vector=style_np)

    methods = {
        'random': shared['random'],
        'full_style': shared['full_style'],
        'mod': shared['mod'],
        'variance': shared['variance'],
    }
    for name, decomp in decompositions.items():
        per = build_steering_vectors(style_np, decomp, rng, mod_vector=style_np)
        methods[f'{name}_h1_dims'] = per['h1_dims']
        methods[f'{name}_h0_removed'] = per['h0_removed']
    return methods


def _summarize_results(method_results, decompositions, seed):
    # Bootstrap CIs per metric per decomposition.
    shared = ['random', 'full_style', 'mod', 'variance']
    summaries = {}
    for dname in decompositions:
        decomp_methods = {sm: method_results[sm] for sm in shared}
        decomp_methods['h1_dims'] = method_results[f'{dname}_h1_dims']
        decomp_methods['h0_removed'] = method_results[f'{dname}_h0_removed']

        per_method = {}
        for mname, results_list in decomp_methods.items():
            valid = [r for r in results_list if r is not None]
            if not valid:
                per_method[mname] = {'n': 0}
                continue
            metrics = {}
            for metric in ('fact', 'subject', 'style', 'coherent', 'joint', 'coherent_joint'):
                values = [r[metric] for r in valid]
                mean, lo, hi = bootstrap_ci(values, seed=seed)
                metrics[metric] = {
                    'mean': round(mean * 100, 1),
                    'ci_lo': round(lo * 100, 1),
                    'ci_hi': round(hi * 100, 1),
                    'raw': values,
                }
            per_method[mname] = {'n': len(valid), 'metrics': metrics}
        summaries[dname] = per_method
    return summaries


def _significance_tests(summaries):
    # McNemar + Mann-Whitney tests for the headline contrasts.
    stats = {}
    for dname, summary in summaries.items():
        per_decomp = {}
        for metric_name in ('joint', 'coherent_joint'):
            for m1 in ('h1_dims', 'h0_removed'):
                for m2 in ('variance', 'random', 'full_style', 'mod'):
                    if summary.get(m1, {}).get('n', 0) == 0:
                        continue
                    if summary.get(m2, {}).get('n', 0) == 0:
                        continue
                    s1 = summary[m1]['metrics'][metric_name]['raw']
                    s2 = summary[m2]['metrics'][metric_name]['raw']
                    n = min(len(s1), len(s2))
                    s1, s2 = s1[:n], s2[:n]
                    p_mcnemar = mcnemar_test(s1, s2)
                    try:
                        _, p_mw = mannwhitneyu(s1, s2, alternative='two-sided')
                    except ValueError:
                        p_mw = 1.0
                    eff = float(np.mean(s1) - np.mean(s2))
                    ratio = float(np.mean(s1) / (np.mean(s2) + 1e-10))
                    per_decomp[f"{m1}_vs_{m2}_{metric_name}"] = {
                        'effect': round(eff, 4),
                        'ratio': round(ratio, 2),
                        'p_mcnemar': round(p_mcnemar, 6),
                        'p_mannwhitney': round(float(p_mw), 6),
                    }
        stats[dname] = per_decomp
    return stats


def _strip_raw(summaries):
    # Drop per-sample raw values before serialization.
    clean = {}
    for dname, summary in summaries.items():
        clean[dname] = {}
        for mname, mdata in summary.items():
            if mdata.get('n', 0) == 0:
                clean[dname][mname] = {'n': 0}
                continue
            clean[dname][mname] = {
                'n': mdata['n'],
                'metrics': {
                    metric: {'mean': vals['mean'], 'ci_lo': vals['ci_lo'], 'ci_hi': vals['ci_hi']}
                    for metric, vals in mdata['metrics'].items()
                },
            }
    return clean


def main():
    parser = argparse.ArgumentParser(description='CounterFact sheaf steering evaluation')
    parser.add_argument('--model', default='mistralai/Mistral-7B-v0.1')
    parser.add_argument('--n_tests', type=int, default=1000)
    parser.add_argument('--n_pairs', type=int, default=200)
    parser.add_argument('--edge_dim', type=int, default=128)
    parser.add_argument('--k', type=int, default=20)
    parser.add_argument('--scale', type=float, default=0.3)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--quantize', default='4bit', choices=['4bit', 'none'])
    parser.add_argument('--layer', type=int, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_new_tokens', type=int, default=100)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    test_data = load_counterfact_data(n_samples=args.n_tests, seed=args.seed)
    mrpc_pairs = load_mrpc_pairs(n_pairs=args.n_pairs, split='validation')

    model, tokenizer = load_causal_model(args.model, args.quantize)
    layer = args.layer if args.layer is not None else int(model_n_layers(model) * 0.625)
    hidden_dim = model.config.hidden_size

    sample_texts = FORMAL_EXAMPLES[:8] + CASUAL_EXAMPLES[:8]
    h_norm_tensor = torch.stack(
        pooled_hidden_states(model, tokenizer, sample_texts, layer, args.batch_size)
    ).norm(dim=-1).mean()
    target_norm = float(h_norm_tensor) * args.scale

    s1_texts = [p[0] for p in mrpc_pairs]
    s2_texts = [p[1] for p in mrpc_pairs]
    bg_texts = list(dict.fromkeys(s1_texts + s2_texts))[:args.n_pairs * 2]

    s1 = torch.stack(pooled_hidden_states(model, tokenizer, s1_texts, layer, args.batch_size)).numpy()
    s2 = torch.stack(pooled_hidden_states(model, tokenizer, s2_texts, layer, args.batch_size)).numpy()
    bg = torch.stack(pooled_hidden_states(model, tokenizer, bg_texts, layer, args.batch_size)).numpy()

    casual = pooled_hidden_states(model, tokenizer, CASUAL_EXAMPLES, layer, args.batch_size)
    formal = pooled_hidden_states(model, tokenizer, FORMAL_EXAMPLES, layer, args.batch_size)
    style_vec = (torch.stack(casual).mean(0) - torch.stack(formal).mean(0)).cpu().numpy()

    decompositions = _build_decompositions(s1, s2, bg, args.edge_dim, args.k)
    methods = _assemble_steering_vectors(decompositions, style_vec, rng)
    scaled = scale_to_norm(methods, target_norm)

    method_results = {name: [] for name in scaled}
    text_outputs = []

    for idx, entry in enumerate(tqdm(test_data, desc='Evaluating')):
        try:
            baseline_text = model_generate(model, tokenizer, entry['gen_prompt'], args.max_new_tokens)
            text_entry = {
                'idx': idx,
                'case_id': entry['case_id'],
                'gen_prompt': entry['gen_prompt'],
                'entity': entry['entity'],
                'subject': entry['subject'],
                'fact': entry['fact'],
                'baseline': baseline_text[:500],
            }
            for mname, vec in scaled.items():
                steered = generate_with_steering(
                    model, tokenizer, entry['gen_prompt'], vec, layer, args.max_new_tokens
                )
                result = evaluate_generation(baseline_text, steered, entry['entity'], entry['subject'])
                method_results[mname].append(result)
                text_entry[f'{mname}_output'] = steered[:500]
                text_entry[f'{mname}_eval'] = result
            text_outputs.append(text_entry)
        except Exception:
            for mname in scaled:
                method_results[mname].append(None)

    summaries = _summarize_results(method_results, decompositions, args.seed)
    statistics = _significance_tests(summaries)

    decomp_meta = {
        dname: {
            'method': decomp['method'],
            'eigval_range': [float(decomp['eigvals'][0]), float(decomp['eigvals'][-1])],
            'h0_dim': int(decomp.get('h0_dim', -1)),
            'spectral_gap': float(decomp.get('spectral_gap', 0)),
            'h1_dims': decomp['h1_dims'],
            'var_h1': decomp['var_h1'],
            'h1_var_overlap': len(set(decomp['h1_dims']) & set(decomp['var_h1'])),
        }
        for dname, decomp in decompositions.items()
    }

    outdir = Path('outputs/counterfact_steering')
    outdir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    slug = args.model.replace('/', '_')

    summary_path = outdir / f'{slug}_{timestamp}.json'
    with open(summary_path, 'w') as f:
        json.dump({
            'model': args.model,
            'layer': layer,
            'd': hidden_dim,
            'h_norm': float(h_norm_tensor),
            'target_steer_norm': target_norm,
            'n_tests': len(test_data),
            'n_pairs_mrpc': len(mrpc_pairs),
            'dataset': 'counterfact',
            'args': vars(args),
            'decompositions': decomp_meta,
            'summaries': _strip_raw(summaries),
            'statistics': statistics,
            'sample_outputs': text_outputs[:50],
        }, f, indent=2, default=str)

    full_path = outdir / f'{slug}_{timestamp}_full_outputs.json'
    with open(full_path, 'w') as f:
        json.dump(text_outputs, f, indent=2, default=str)

    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
