#!/usr/bin/env python3
"""
Pooling sensitivity for sheaf H^0 retrieval.

Compares three pooling variants (mean over non-pad tokens, last non-pad
token, first token) using the same H^0 retrieval pipeline. Sheaf H^0 beats
the matched Full hidden state at every pooling choice; absolute numbers
depend on whether the pooling captures content.

Output: outputs/pooling_sensitivity/{model}_{timestamp}.json
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import torch

from sheafint import fit_h0_pca
from utils.datasets import (
    build_centroid_query_split,
    hard_restrict_by_relation,
    load_counterfact_with_paraphrases,
    split_facts_train_test,
)
from utils.model_io import load_causal_model, n_layers
from utils.retrieval import pooled_hidden_states_variant, retrieval_with_basis

from run_leace_counterfact import make_train_pairs


def run_one_pooling(model, tokenizer, layer_idx, pooling, batch_size,
                    train_a, train_b, test_split, hard_restrict,
                    k, edge_dim):
    H_train_a = pooled_hidden_states_variant(
        model, tokenizer, train_a, layer_idx, pooling=pooling,
        batch_size=batch_size,
    )
    H_train_b = pooled_hidden_states_variant(
        model, tokenizer, train_b, layer_idx, pooling=pooling,
        batch_size=batch_size,
    )
    H_cent = pooled_hidden_states_variant(
        model, tokenizer, test_split['centroid_prompts'], layer_idx,
        pooling=pooling, batch_size=batch_size,
    ).numpy()
    H_qry = pooled_hidden_states_variant(
        model, tokenizer, test_split['query_prompts'], layer_idx,
        pooling=pooling, batch_size=batch_size,
    ).numpy()

    h0_basis = fit_h0_pca(H_train_a, H_train_b, k=k, edge_dim=edge_dim)
    cf_id = test_split['centroid_fact_id']
    q_id = test_split['query_fact_id']

    def score(basis, restrict):
        return retrieval_with_basis(
            H_qry, H_cent, basis, cf_id, q_id, restrict_to=restrict,
        )

    return {
        'Full_hard': score(None, hard_restrict),
        'H0_sheaf_hard': score(h0_basis, hard_restrict),
        'Full_easy': score(None, None),
        'H0_sheaf_easy': score(h0_basis, None),
    }


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--n_facts', type=int, default=500)
    ap.add_argument('--poolings', nargs='+', default=['mean', 'last', 'first'])
    ap.add_argument('--train_frac', type=float, default=0.70)
    ap.add_argument('--layer_frac', type=float, default=0.67)
    ap.add_argument('--k', type=int, default=20)
    ap.add_argument('--edge_dim', type=int, default=128)
    ap.add_argument('--quantize', default='4bit', choices=['none', '4bit'])
    ap.add_argument('--output_dir', default='outputs/pooling_sensitivity')
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
    hard_restrict = hard_restrict_by_relation(
        facts, test_ids, test_split['query_relation_id'],
    )

    model, tokenizer = load_causal_model(args.model, quantize=args.quantize)
    layer_idx = int(args.layer_frac * n_layers(model))
    d = model.config.hidden_size

    results = {}
    for pooling in args.poolings:
        results[pooling] = run_one_pooling(
            model, tokenizer, layer_idx, pooling, args.batch_size,
            train_a, train_b, test_split, hard_restrict,
            args.k, args.edge_dim,
        )

    del model
    torch.cuda.empty_cache()

    payload = {
        'model': args.model,
        'hidden_dim': d,
        'layer': layer_idx,
        'n_train_facts': len(train_ids),
        'n_test_facts': len(test_ids),
        'k': args.k,
        'edge_dim': args.edge_dim,
        'poolings': list(args.poolings),
        'results_by_pooling': results,
        'seed': args.seed,
        'timestamp': ts,
    }
    out_path.write_text(json.dumps(payload, indent=2))


if __name__ == '__main__':
    main()
