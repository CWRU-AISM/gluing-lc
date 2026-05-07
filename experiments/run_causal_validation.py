#!/usr/bin/env python3
"""
Causal validation of H^0 / H^1 dimensions (Table 1).

Compares H^1 ablations to variance-matched dimension controls and reports
bootstrap CIs on the resulting effect ratios. The variance match is
critical because raw effect sizes scale with the total variance ablated.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
import numpy as np
from tqdm import tqdm
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.utils.causal import (
    ablation_effect,
    bootstrap_effect_ratio,
    identify_h0_h1_dims,
    variance_matched_dims,
)
from experiments.utils.datasets import load_mrpc_pairs
from experiments.utils.model_io import load_causal_model, n_layers as model_n_layers


def _to_jsonable(obj):
    # Convert numpy scalars / arrays into JSON-friendly Python types.
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_jsonable(v) for v in obj]
    return obj


def _run_layer(args, model, tokenizer, layer, paraphrase_pairs, all_texts):
    # Run ablation experiment for a single layer.
    h0_dims, h1_dims, variance = identify_h0_h1_dims(
        model, tokenizer, paraphrase_pairs[:50], layer, n_dims=20,
    )
    if not h0_dims or not h1_dims:
        return None

    h1_variance = float(variance[h1_dims].sum())
    random_dims = np.random.choice(model.config.hidden_size, size=20, replace=False).tolist()
    vm_dims = variance_matched_dims(
        target_variance=h1_variance,
        variance_per_dim=variance,
        n_dims=20,
        exclude_dims=h0_dims + h1_dims,
    )

    n_total = model_n_layers(model)
    conditions = {'h0': h0_dims, 'h1': h1_dims, 'random': random_dims, 'variance_matched': vm_dims}
    results = {name: [] for name in conditions}

    for text in tqdm(all_texts, desc=f'Ablation L{layer}'):
        for name, dims in conditions.items():
            try:
                effect = ablation_effect(model, tokenizer, text, layer, n_total, dims)
                if not np.isnan(effect['l2_distance']):
                    results[name].append(effect)
            except Exception:
                continue

    by_l2 = {name: [r['l2_distance'] for r in results[name]] for name in conditions}
    return {
        'h0_dims': h0_dims,
        'h1_dims': h1_dims,
        'h0_variance': float(variance[h0_dims].sum()),
        'h1_variance': h1_variance,
        'vm_variance': float(variance[vm_dims].sum()),
        'h0_effect': {'mean': np.mean(by_l2['h0']), 'std': np.std(by_l2['h0'])},
        'h1_effect': {'mean': np.mean(by_l2['h1']), 'std': np.std(by_l2['h1'])},
        'random_effect': {'mean': np.mean(by_l2['random']), 'std': np.std(by_l2['random'])},
        'vm_effect': {'mean': np.mean(by_l2['variance_matched']), 'std': np.std(by_l2['variance_matched'])},
        'h1_vs_vm': bootstrap_effect_ratio(by_l2['h1'], by_l2['variance_matched'], args.n_bootstrap),
        'h1_vs_h0': bootstrap_effect_ratio(by_l2['h1'], by_l2['h0'], args.n_bootstrap),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='gpt2')
    parser.add_argument('--quantize', type=str, default='none')
    parser.add_argument('--n_samples', type=int, default=100)
    parser.add_argument('--n_bootstrap', type=int, default=1000)
    args = parser.parse_args()

    model, tokenizer = load_causal_model(args.model, args.quantize)
    paraphrase_pairs = load_mrpc_pairs(n_pairs=100)
    all_texts = [s for pair in paraphrase_pairs for s in pair][:args.n_samples]

    n_total = model_n_layers(model)
    layers_to_test = [n_total // 4, n_total // 2, 3 * n_total // 4]
    experiments = {}
    for layer in layers_to_test:
        result = _run_layer(args, model, tokenizer, layer, paraphrase_pairs, all_texts)
        if result is not None:
            experiments[f'layer_{layer}'] = result

    output_dir = Path('outputs/causal_validation')
    output_dir.mkdir(parents=True, exist_ok=True)
    slug = args.model.replace('/', '_')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = output_dir / f'{slug}_{timestamp}.json'

    with open(output_file, 'w') as f:
        json.dump(_to_jsonable({
            'model': args.model,
            'n_samples': len(all_texts),
            'experiments': experiments,
        }), f, indent=2)


if __name__ == '__main__':
    main()
