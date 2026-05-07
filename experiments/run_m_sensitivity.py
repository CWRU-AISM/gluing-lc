#!/usr/bin/env python3
"""
H^0 dimension (m) sensitivity sweep.

Sweeps m in {5, 10, 20, 40, 80, 128} and reports hard CounterFact retrieval
accuracy per m using the same pipeline as run_leace_dim_controlled. The
boundary case m = edge_dim has no kernel selection and serves as a control.

Output: outputs/m_sensitivity/{model}_{timestamp}.json
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from sheafint import fit_h0_pca
from utils.datasets import (
    build_centroid_query_split,
    hard_restrict_by_relation,
    load_counterfact_with_paraphrases,
    split_facts_train_test,
)
from utils.model_io import load_causal_model, n_layers, pooled_hidden_states
from utils.retrieval import retrieval_with_basis

from run_leace_counterfact import make_train_pairs, stack_pooled


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--n_facts', type=int, default=500)
    ap.add_argument('--m_values', type=int, nargs='+',
                    default=[5, 10, 20, 40, 80, 128])
    ap.add_argument('--train_frac', type=float, default=0.70)
    ap.add_argument('--layer_frac', type=float, default=0.67)
    ap.add_argument('--edge_dim', type=int, default=128)
    ap.add_argument('--quantize', default='4bit', choices=['none', '4bit'])
    ap.add_argument('--output_dir', default='outputs/m_sensitivity')
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

    hard_restrict = hard_restrict_by_relation(
        facts, test_ids, test_split['query_relation_id'],
    )
    centroid_fact_id = test_split['centroid_fact_id']
    query_fact_id = test_split['query_fact_id']

    def score(basis, restrict):
        return retrieval_with_basis(
            H_qry, H_cent, basis, centroid_fact_id, query_fact_id,
            restrict_to=restrict,
        )

    hard_lookup = {}
    easy_lookup = {}
    for m in args.m_values:
        m_clip = min(m, args.edge_dim)
        basis = fit_h0_pca(H_train_a, H_train_b, k=m_clip, edge_dim=args.edge_dim)
        hard_lookup[str(m)] = score(basis, hard_restrict)
        easy_lookup[str(m)] = score(basis, None)

    payload = {
        'model': args.model,
        'hidden_dim': d,
        'layer': layer_idx,
        'n_train_facts': len(train_ids),
        'n_test_facts': len(test_ids),
        'edge_dim': args.edge_dim,
        'm_values': list(args.m_values),
        'hard_lookup_by_m': hard_lookup,
        'easy_lookup_by_m': easy_lookup,
        'hard_lookup_full_dim': score(None, hard_restrict),
        'easy_lookup_full_dim': score(None, None),
        'seed': args.seed,
        'timestamp': ts,
    }
    out_path.write_text(json.dumps(payload, indent=2))


if __name__ == '__main__':
    main()
