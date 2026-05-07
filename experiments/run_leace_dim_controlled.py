#!/usr/bin/env python3
"""
Dimension-controlled LEACE comparison.

LEACE for a binary concept is a rank-1 eraser, so its preserved subspace is
(d-1)-dim; the matched 20-D baseline is the top-20 PCs of that preserved
subspace. We add this explicit comparison alongside sheaf H^0 (20D), the
full-dim LEACE-preserved retrieval, PCA-20, and Random-20 baselines on hard
same-relation CounterFact retrieval.

Output: outputs/leace_dim_controlled/{model}_{timestamp}.json
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

from run_leace_counterfact import (
    fit_leace_eraser,
    fit_pca_basis,
    fit_random_bases,
    make_train_pairs,
    stack_pooled,
)


def fit_leace_20d_basis(H_leace_preserved: np.ndarray, k: int = 20) -> np.ndarray:
    """
    Top-k PCs of LEACE-preserved activations.

    Used as the dim-controlled baseline against the bottom-k eigenvectors of
    the sheaf Laplacian: same projection rank, different objective.
    """
    Xc = H_leace_preserved - H_leace_preserved.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    basis = Vt[:k].T
    basis, _ = np.linalg.qr(basis)
    return basis


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--n_facts', type=int, default=500)
    ap.add_argument('--train_frac', type=float, default=0.70)
    ap.add_argument('--layer_frac', type=float, default=0.67)
    ap.add_argument('--k', type=int, default=20)
    ap.add_argument('--edge_dim', type=int, default=128)
    ap.add_argument('--quantize', default='4bit', choices=['none', '4bit'])
    ap.add_argument('--output_dir', default='outputs/leace_dim_controlled')
    ap.add_argument('--n_random_trials', type=int, default=5)
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
    H_cent_t = stack_pooled(pooled_hidden_states(
        model, tokenizer, test_split['centroid_prompts'], layer_idx,
        batch_size=args.batch_size,
    ))
    H_qry_t = stack_pooled(pooled_hidden_states(
        model, tokenizer, test_split['query_prompts'], layer_idx,
        batch_size=args.batch_size,
    ))
    H_cent = H_cent_t.numpy()
    H_qry = H_qry_t.numpy()

    del model
    torch.cuda.empty_cache()

    h0_basis = fit_h0_pca(H_train_a, H_train_b, k=args.k, edge_dim=args.edge_dim)
    eraser = fit_leace_eraser(H_train_a, args.seed)
    H_train_a_leace = eraser(H_train_a).numpy()
    H_cent_leace = eraser(H_cent_t).numpy()
    H_qry_leace = eraser(H_qry_t).numpy()
    leace_20d_basis = fit_leace_20d_basis(H_train_a_leace, k=args.k)
    pca_basis = fit_pca_basis(H_train_a, H_train_b, args.k)
    rand_bases = fit_random_bases(d, args.k, args.n_random_trials, args.seed)

    hard_restrict = hard_restrict_by_relation(
        facts, test_ids, test_split['query_relation_id'],
    )
    centroid_fact_id = test_split['centroid_fact_id']
    query_fact_id = test_split['query_fact_id']

    def score(basis, restrict, query=H_qry, cent=H_cent):
        return retrieval_with_basis(
            query, cent, basis, centroid_fact_id, query_fact_id,
            restrict_to=restrict,
        )

    hard_metrics = {
        'Full': score(None, hard_restrict),
        'H0_sheaf_20d': score(h0_basis, hard_restrict),
        'LEACE_preserved_full': score(None, hard_restrict,
                                      query=H_qry_leace, cent=H_cent_leace),
        'LEACE_preserved_20d': score(leace_20d_basis, hard_restrict,
                                      query=H_qry_leace, cent=H_cent_leace),
        'PCA_top20': score(pca_basis, hard_restrict),
        'Random_20d': float(np.mean([
            score(b, hard_restrict) for b in rand_bases
        ])),
    }
    easy_metrics = {
        'Full': score(None, None),
        'H0_sheaf_20d': score(h0_basis, None),
        'LEACE_preserved_full': score(None, None,
                                      query=H_qry_leace, cent=H_cent_leace),
        'LEACE_preserved_20d': score(leace_20d_basis, None,
                                      query=H_qry_leace, cent=H_cent_leace),
        'PCA_top20': score(pca_basis, None),
        'Random_20d': float(np.mean([score(b, None) for b in rand_bases])),
    }

    payload = {
        'model': args.model,
        'hidden_dim': d,
        'layer': layer_idx,
        'n_train_facts': len(train_ids),
        'n_test_facts': len(test_ids),
        'n_train_pairs': int(H_train_a.shape[0]),
        'k': args.k,
        'edge_dim': args.edge_dim,
        'easy_lookup_held_out': easy_metrics,
        'hard_lookup_held_out': hard_metrics,
        'n_random_trials': args.n_random_trials,
        'seed': args.seed,
        'timestamp': ts,
    }
    out_path.write_text(json.dumps(payload, indent=2))


if __name__ == '__main__':
    main()
