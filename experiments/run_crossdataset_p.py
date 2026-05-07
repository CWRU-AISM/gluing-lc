#!/usr/bin/env python3
"""
Cross-dataset restriction-map transfer.

Tests whether the framework requires same-distribution paraphrases or
transfers across paraphrase styles. We fit the joint-PCA P on out-of-domain
paraphrase pairs (MRPC, PAWS, QQP) and use it for hard-CounterFact
retrieval. Compared against in-domain CounterFact P, PCA-20, and Random-20.

Output: outputs/crossdataset_p/{model}_{timestamp}.json
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
    load_paraphrase_pairs,
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
    ap.add_argument('--n_paraphrase_pairs', type=int, default=300)
    ap.add_argument('--datasets', nargs='+',
                    default=['mrpc', 'paws', 'qqp'])
    ap.add_argument('--train_frac', type=float, default=0.70)
    ap.add_argument('--layer_frac', type=float, default=0.67)
    ap.add_argument('--k', type=int, default=20)
    ap.add_argument('--edge_dim', type=int, default=128)
    ap.add_argument('--quantize', default='4bit', choices=['none', '4bit'])
    ap.add_argument('--output_dir', default='outputs/crossdataset_p')
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
    cf_a, cf_b = make_train_pairs(facts, train_ids)

    od_pairs = {
        name: load_paraphrase_pairs(name, args.n_paraphrase_pairs, seed=args.seed)
        for name in args.datasets
    }

    model, tokenizer = load_causal_model(args.model, quantize=args.quantize)
    layer_idx = int(args.layer_frac * n_layers(model))
    d = model.config.hidden_size

    H_cf_a = stack_pooled(pooled_hidden_states(
        model, tokenizer, cf_a, layer_idx, batch_size=args.batch_size,
    ))
    H_cf_b = stack_pooled(pooled_hidden_states(
        model, tokenizer, cf_b, layer_idx, batch_size=args.batch_size,
    ))

    od_acts = {}
    for name, pairs in od_pairs.items():
        a_prompts = [p[0] for p in pairs]
        b_prompts = [p[1] for p in pairs]
        H_a = stack_pooled(pooled_hidden_states(
            model, tokenizer, a_prompts, layer_idx, batch_size=args.batch_size,
        ))
        H_b = stack_pooled(pooled_hidden_states(
            model, tokenizer, b_prompts, layer_idx, batch_size=args.batch_size,
        ))
        od_acts[name] = (H_a, H_b)

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

    bases = {
        'H0_counterfact': fit_h0_pca(
            H_cf_a, H_cf_b, k=args.k, edge_dim=args.edge_dim,
        ),
    }
    for name, (H_a, H_b) in od_acts.items():
        bases[f'H0_{name}'] = fit_h0_pca(
            H_a, H_b, k=args.k, edge_dim=args.edge_dim,
        )
    bases['PCA_top20'] = fit_pca_basis(H_cf_a, H_cf_b, args.k)
    rand_bases = fit_random_bases(d, args.k, 5, args.seed)

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

    hard_metrics = {'Full': score(None)}
    for name, basis in bases.items():
        hard_metrics[name] = score(basis)
    rand_scores = [score(b) for b in rand_bases]
    hard_metrics['Random_20_mean'] = float(np.mean(rand_scores))
    hard_metrics['Random_20_std'] = float(np.std(rand_scores))

    payload = {
        'model': args.model,
        'hidden_dim': d,
        'layer': layer_idx,
        'n_test_facts': len(test_ids),
        'n_paraphrase_pairs_per_source': args.n_paraphrase_pairs,
        'datasets_loaded': list(od_acts.keys()),
        'k': args.k,
        'hard_lookup_held_out': hard_metrics,
        'seed': args.seed,
        'timestamp': ts,
    }
    out_path.write_text(json.dumps(payload, indent=2))


if __name__ == '__main__':
    main()
