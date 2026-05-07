#!/usr/bin/env python3
"""
Cycle-aware H^0 retrieval: multi-paraphrase rings beat the pair-only forest.

The pair-only paraphrase graph is acyclic (b_1 = 0) and the kernel of
L_sheaf reduces to within-pair covariance. Multi-paraphrase rings
(b_1 = N per fact) and full cliques (b_1 = N(K-1)(K-2)/2) require
invariance across all paraphrases of each fact, which produces strictly
stronger H^0 directions for hard same-relation retrieval.

Output: outputs/cycle_h0/{model}_{timestamp}.json
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from sheafint import (
    clique_edges,
    count_clique_cycles,
    count_ring_cycles,
    fit_h0_from_edges,
    pair_edges,
    ring_edges,
)
from utils.datasets import (
    build_centroid_query_split,
    hard_restrict_by_relation,
    load_counterfact_with_paraphrases,
    split_facts_train_test,
)
from utils.model_io import load_causal_model, n_layers, pooled_hidden_states
from utils.retrieval import retrieval_with_basis

from run_leace_counterfact import fit_pca_basis, stack_pooled


def collect_node_prompts(facts: list, train_ids: List[int],
                          node_names: List[str]) -> List[str]:
    name_to_prompt: Dict[str, str] = {}
    for i in train_ids:
        for j, expr in enumerate(facts[i]['expressions']):
            name_to_prompt[f'f{i}_p{j}'] = expr
    return [name_to_prompt[n] for n in node_names]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--n_facts', type=int, default=500)
    ap.add_argument('--train_frac', type=float, default=0.70)
    ap.add_argument('--layer_frac', type=float, default=0.67)
    ap.add_argument('--k', type=int, default=20)
    ap.add_argument('--edge_dim', type=int, default=128)
    ap.add_argument('--quantize', default='4bit', choices=['none', '4bit'])
    ap.add_argument('--output_dir', default='outputs/cycle_h0')
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

    pair_e, pair_nodes = pair_edges(facts, train_ids)
    ring_e, ring_nodes = ring_edges(facts, train_ids)
    clique_e, clique_nodes = clique_edges(facts, train_ids)
    all_nodes = sorted(set(pair_nodes + ring_nodes + clique_nodes))
    train_prompts = collect_node_prompts(facts, train_ids, all_nodes)

    model, tokenizer = load_causal_model(args.model, quantize=args.quantize)
    layer_idx = int(args.layer_frac * n_layers(model))
    d = model.config.hidden_size

    H_train = stack_pooled(pooled_hidden_states(
        model, tokenizer, train_prompts, layer_idx, batch_size=args.batch_size,
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

    activations = {n: H_train[i].numpy() for i, n in enumerate(all_nodes)}
    h0_pair = fit_h0_from_edges(activations, pair_e, k=args.k, edge_dim=args.edge_dim)
    h0_ring = fit_h0_from_edges(activations, ring_e, k=args.k, edge_dim=args.edge_dim)
    h0_clique = fit_h0_from_edges(activations, clique_e, k=args.k, edge_dim=args.edge_dim)

    pair_acts = torch.from_numpy(np.stack([activations[n] for n in pair_nodes]))
    pca_basis = fit_pca_basis(pair_acts, pair_acts, args.k)

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

    hard_metrics = {
        'Full': score(None),
        'H0_pair_only': score(h0_pair),
        'H0_multi_paraphrase_ring': score(h0_ring),
        'H0_multi_paraphrase_clique': score(h0_clique),
        'PCA_top20': score(pca_basis),
    }

    payload = {
        'model': args.model,
        'hidden_dim': d,
        'layer': layer_idx,
        'n_train_facts': len(train_ids),
        'n_test_facts': len(test_ids),
        'n_pair_edges': len(pair_e),
        'n_ring_edges': len(ring_e),
        'n_clique_edges': len(clique_e),
        'b1_ring': count_ring_cycles(facts, train_ids),
        'b1_clique': count_clique_cycles(facts, train_ids),
        'k': args.k,
        'edge_dim': args.edge_dim,
        'hard_lookup_held_out': hard_metrics,
        'seed': args.seed,
        'timestamp': ts,
    }
    out_path.write_text(json.dumps(payload, indent=2))


if __name__ == '__main__':
    main()
