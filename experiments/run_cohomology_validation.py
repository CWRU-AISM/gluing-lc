#!/usr/bin/env python3
# Cross-check the variance heuristic against actual sheaf cohomology and Laplacian spectrum.

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import torch
from tqdm import tqdm
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sheafint import ScalableSheaf
from experiments.utils.causal import hidden_states_at
from experiments.utils.datasets import load_mrpc_pairs
from experiments.utils.model_io import load_causal_model, n_layers as model_n_layers


def _variance_dim_split(
    pairs: List[Tuple[str, str]],
    model,
    tokenizer,
    layer: int,
    k: int,
) -> Dict:
    # Variance-based H0 / H1 candidate dims plus stacked activations for cohomology.
    diffs = []
    activations_a = []
    activations_b = []
    for s1, s2 in tqdm(pairs, desc='Variance'):
        try:
            h_a = hidden_states_at(model, tokenizer, s1, layer).mean(dim=0)
            h_b = hidden_states_at(model, tokenizer, s2, layer).mean(dim=0)
            diffs.append((h_a - h_b).abs().cpu().float())
            activations_a.append(h_a.cpu().float().numpy())
            activations_b.append(h_b.cpu().float().numpy())
        except Exception:
            continue

    mean_diff = torch.stack(diffs).mean(dim=0).numpy()
    sorted_dims = np.argsort(mean_diff)
    return {
        'h0_dims': sorted_dims[:k].tolist(),
        'h1_dims': sorted_dims[-k:][::-1].tolist(),
        'variance_by_dim': mean_diff,
        'activations_a': np.stack(activations_a),
        'activations_b': np.stack(activations_b),
    }


def _sheaf_diagnostics(activations_a: np.ndarray, activations_b: np.ndarray, device: str) -> Dict:
    # Fit a ScalableSheaf and report cohomology, spectrum, and H0 projection diagnostics.
    n_samples = activations_a.shape[0]
    sheaf = ScalableSheaf(
        edge_dim=min(64, activations_a.shape[1], n_samples - 1),
        face_dim=32,
        svd_rank=min(50, n_samples - 1),
        device=device,
    )
    features = {
        'context_a': torch.tensor(activations_a, dtype=torch.float32),
        'context_b': torch.tensor(activations_b, dtype=torch.float32),
    }
    pair_overlaps = {('context_a', 'context_b'): torch.arange(n_samples)}
    sheaf.fit(features, pair_overlaps, method='joint_pca')

    return {
        'cohomology': sheaf.compute_cohomology(),
        'spectrum': sheaf.compute_laplacian_spectrum(k=50),
        'h0_projection': sheaf.project_onto_cohomology(features),
    }


def _summarize_correlation(variance_result: Dict, diagnostics: Dict) -> Dict:
    # Summary statistics relating the variance heuristic to actual cohomology.
    variance = variance_result['variance_by_dim']
    h0_set = set(variance_result['h0_dims'])
    h1_set = set(variance_result['h1_dims'])
    h0_var = float(np.mean([variance[d] for d in h0_set]))
    h1_var = float(np.mean([variance[d] for d in h1_set]))
    return {
        'h0_dim_actual': diagnostics['spectrum']['h0_dim'],
        'h1_dim_actual': diagnostics['cohomology']['H1_dim'],
        'spectral_gap': diagnostics['spectrum']['spectral_gap'],
        'h0_fraction': diagnostics['h0_projection']['h0_fraction'],
        'coboundary_check': diagnostics['h0_projection']['coboundary_of_h0_projection'],
        'is_valid_projection': diagnostics['h0_projection']['is_valid'],
        'variance_h0_mean': h0_var,
        'variance_h1_mean': h1_var,
        'variance_ratio': h1_var / (h0_var + 1e-10),
    }


def main():
    parser = argparse.ArgumentParser(description='Cohomology vs variance validation')
    parser.add_argument('--model', type=str, default='gpt2')
    parser.add_argument('--quantize', type=str, default='none')
    parser.add_argument('--n_pairs', type=int, default=100)
    parser.add_argument('--k', type=int, default=64)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tokenizer = load_causal_model(args.model, args.quantize)
    layer = model_n_layers(model) * 2 // 3

    pairs = load_mrpc_pairs(n_pairs=args.n_pairs, split='validation')
    variance_result = _variance_dim_split(pairs, model, tokenizer, layer, args.k)
    diagnostics = _sheaf_diagnostics(
        variance_result['activations_a'], variance_result['activations_b'], device,
    )
    correlation = _summarize_correlation(variance_result, diagnostics)

    output_dir = Path('outputs/cohomology_validation')
    output_dir.mkdir(parents=True, exist_ok=True)
    slug = args.model.replace('/', '_')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_file = output_dir / f'{slug}_{timestamp}.json'

    with open(out_file, 'w') as f:
        json.dump({
            'model': args.model,
            'layer': layer,
            'n_pairs': len(pairs),
            'timestamp': datetime.now().isoformat(),
            'variance_h0_dims': variance_result['h0_dims'][:20],
            'variance_h1_dims': variance_result['h1_dims'][:20],
            'cohomology': diagnostics['cohomology'],
            'spectrum': {
                'h0_dim': diagnostics['spectrum']['h0_dim'],
                'spectral_gap': diagnostics['spectrum']['spectral_gap'],
                'effective_rank': diagnostics['spectrum']['effective_rank'],
            },
            'h0_projection': {
                'h0_fraction': diagnostics['h0_projection']['h0_fraction'],
                'is_valid': diagnostics['h0_projection']['is_valid'],
            },
            'correlation': correlation,
        }, f, indent=2)


if __name__ == '__main__':
    main()
