#!/usr/bin/env python3
"""
Restriction-map ablation for sheaf H^0 retrieval.

Proposition 3.1 reduces L_sheaf to projected within-pair covariance on the
matching graph, raising the worry that joint-PCA P is doing the work and
the kernel selection is decorative. This script tests three regimes:

* PCA-P: joint-PCA top-128 PCs (current method).
* Random-P: random orthonormal d x 128 (5 seeds).
* Identity-P: no projection; bottom-20 eigenvectors of C_W in full d-space.

Output: outputs/restriction_ablation/{model}_{timestamp}.json
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from sheafint import fit_h0_identity, fit_h0_pca, fit_h0_random
from utils.datasets import (
    build_centroid_query_split,
    hard_restrict_by_relation,
    load_counterfact_with_paraphrases,
    split_facts_train_test,
)
from utils.model_io import load_causal_model, n_layers, pooled_hidden_states
from utils.retrieval import retrieval_with_basis

from run_leace_counterfact import (
    fit_pca_basis,
    fit_random_bases,
    make_train_pairs,
    stack_pooled,
)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--n_facts', type=int, default=500)
    ap.add_argument('--train_frac', type=float, default=0.70)
    ap.add_argument('--layer_frac', type=float, default=0.67)
    ap.add_argument('--k', type=int, default=20)
    ap.add_argument('--edge_dim', type=int, default=128)
    ap.add_argument('--n_random_trials', type=int, default=5)
    ap.add_argument('--quantize', default='4bit', choices=['none', '4bit'])
    ap.add_argument('--output_dir', default='outputs/restriction_ablation')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--batch_size', type=int, default=8)
    return ap.parse_args()


def main():
    args = parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_model = args.model.replace('/', '_')
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = out_dir / f'{safe_model}_{ts}.json'

    facts = load_counterfact_with_paraphrases(
        n_facts=args.n_facts, min_expressions=4, seed=args.seed,
    )
    train_ids, test_ids = split_facts_train_test(
        facts, train_frac=args.train_frac, seed=args.seed,
    )
    test_split = build_centroid_query_split(facts, test_ids, seed=args.seed)
    train_a, train_b = make_train_pairs(facts, train_ids)

    model, tokenizer = load_causal_model(args.model, quantize=args.quantize)
    layer_idx = int(args.layer_frac * n_layers(model))
    d = model.config.hidden_size

    H_train_a = stack_pooled(pooled_hidden_states(
        model, tokenizer, train_a, layer_idx, batch_size=args.batch_size,
    ))
    H_train_b = stack_pooled(pooled_hidden_states(
        model, tokenizer, train_b, layer_idx, batch_size=args.batch_size,
    ))
    H_cent = stack_pooled(pooled_hidden_states(
        model, tokenizer, test_split['centroid_prompts'], layer_idx,
        batch_size=args.batch_size,
    )).numpy()
    H_qry = stack_pooled(pooled_hidden_states(
        model, tokenizer, test_split['query_prompts'], layer_idx,
        batch_size=args.batch_size,
    )).numpy()

    del model
    torch.cuda.empty_cache()

    h0_pca = fit_h0_pca(H_train_a, H_train_b, k=args.k, edge_dim=args.edge_dim)
    h0_identity = fit_h0_identity(H_train_a, H_train_b, k=args.k)
    h0_random_bases = [
        fit_h0_random(H_train_a, H_train_b, k=args.k,
                      edge_dim=args.edge_dim, seed=args.seed + 100 + s)
        for s in range(args.n_random_trials)
    ]
    pca_basis = fit_pca_basis(H_train_a, H_train_b, args.k)
    rand_bases = fit_random_bases(d, args.k, args.n_random_trials, args.seed)

    hard_restrict = hard_restrict_by_relation(
        facts, test_ids, test_split['query_relation_id'],
    )
    centroid_fact_id = test_split['centroid_fact_id']
    query_fact_id = test_split['query_fact_id']

    def score(basis):
        return retrieval_with_basis(
            H_qry, H_cent, basis, centroid_fact_id, query_fact_id,
            restrict_to=hard_restrict,
        )

    h0_random_scores = [score(b) for b in h0_random_bases]
    rand_20_scores = [score(b) for b in rand_bases]

    hard_metrics = {
        'Full': score(None),
        'H0_PCA_P': score(h0_pca),
        'H0_random_P_mean': float(np.mean(h0_random_scores)),
        'H0_random_P_std': float(np.std(h0_random_scores)),
        'H0_identity_P': score(h0_identity),
        'PCA_20': score(pca_basis),
        'Random_20_mean': float(np.mean(rand_20_scores)),
        'Random_20_std': float(np.std(rand_20_scores)),
    }

    payload = {
        'model': args.model,
        'hidden_dim': d,
        'layer': layer_idx,
        'n_train_facts': len(train_ids),
        'n_test_facts': len(test_ids),
        'k': args.k,
        'edge_dim': args.edge_dim,
        'n_random_trials': args.n_random_trials,
        'hard_lookup_held_out': hard_metrics,
        'seed': args.seed,
        'timestamp': ts,
    }
    out_path.write_text(json.dumps(payload, indent=2))


if __name__ == '__main__':
    main()
