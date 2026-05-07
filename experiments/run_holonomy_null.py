#!/usr/bin/env python3
"""
Null calibration for relation-hypergraph holonomy obstructions.

Cycle holonomies on small-N Procrustes alignment naturally approach the
orthogonal upper bound sqrt(2k) regardless of structure. This script
calibrates the observed obstruction against:

* a shuffled-paraphrase null (refit P_v + Q_ij after permuting which
  paraphrase belongs to which fact within its relation),
* a tokens-per-node bootstrap (refit with N in {2, 3, 4, 5, 6}),
* an identity-transport sanity anchor.

Output: outputs/holonomy_null/{model}_{timestamp}.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from sheafint import (
    cycle_obstructions,
    find_fundamental_cycles,
    fit_per_node_pca,
    fit_procrustes_transports,
)
from sheafint.core.hypergraph import HyperedgeComplex
from sheafint.data import build_facts


def build_relation_hypergraph(facts, n_per_fact=5):
    """
    Build the relation hypergraph for the holonomy null calibration.

    Each fact contributes one node per paraphrase joined in a ring; facts
    sharing a relation are themselves chained in a per-relation ring through
    their first paraphrase. The result has both within-fact and across-fact
    fundamental cycles.
    """
    H = HyperedgeComplex(mode='ring')
    rel_to_facts: Dict[str, List[int]] = defaultdict(list)
    for i, fact in enumerate(facts):
        members = [
            f'f{i}_p{j}'
            for j in range(min(n_per_fact, len(fact['paraphrases'])))
        ]
        H.add_cluster(members)
        rel_to_facts[fact['relation_id']].append(i)
    for rel, fact_ids in rel_to_facts.items():
        if len(fact_ids) < 2:
            continue
        ring = [f'f{i}_p0' for i in fact_ids]
        for a, b in zip(ring, ring[1:] + ring[:1]):
            H._add_edge(a, b)
    return H, rel_to_facts


@torch.no_grad()
def extract_token_activations(model, tokenizer, prompt, layer, device, n_last=6):
    ids = tokenizer(
        prompt, return_tensors='pt', truncation=True, max_length=64,
    ).to(device)
    h = model(**ids, output_hidden_states=True).hidden_states[layer]
    h = h.squeeze(0)
    take = min(n_last, h.shape[0])
    return h[-take:].float().cpu()


def shuffle_paraphrases_within_relation(
    node_features: Dict[str, torch.Tensor],
    rel_to_facts: Dict[str, List[int]],
    n_per_fact: int,
    seed: int,
) -> Dict[str, torch.Tensor]:
    rng = np.random.default_rng(seed)
    new_features: Dict[str, torch.Tensor] = {}
    for fact_ids in rel_to_facts.values():
        node_names: List[str] = []
        feature_pool: List[torch.Tensor] = []
        for fid in fact_ids:
            for p in range(n_per_fact):
                name = f'f{fid}_p{p}'
                if name in node_features:
                    node_names.append(name)
                    feature_pool.append(node_features[name])
        perm = rng.permutation(len(node_names))
        for j, name in enumerate(node_names):
            new_features[name] = feature_pool[perm[j]]
    for name, feat in node_features.items():
        new_features.setdefault(name, feat)
    return new_features


def load_quantized(model_name: str, quantize: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if quantize == '4bit':
        config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type='nf4',
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=config, device_map='auto',
            torch_dtype=torch.float16,
        )
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map=device,
        )
    model.training = False
    return model, tokenizer


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='gpt2')
    ap.add_argument('--layer', type=int, default=8)
    ap.add_argument('--edge_dim', type=int, default=8)
    ap.add_argument('--n_per_fact', type=int, default=5)
    ap.add_argument('--n_token_samples', type=int, default=6)
    ap.add_argument('--n_shuffle_seeds', type=int, default=20)
    ap.add_argument('--bootstrap_n_values', type=int, nargs='+',
                    default=[2, 3, 4, 5, 6])
    ap.add_argument('--quantize', default='none', choices=['none', '4bit'])
    ap.add_argument('--output_dir', default='outputs/holonomy_null')
    ap.add_argument('--seed', type=int, default=0)
    return ap.parse_args()


def main():
    args = parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_model = args.model.replace('/', '_')
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = out_dir / f'{safe_model}_{ts}.json'

    facts = build_facts(n_relations=6, paraphrases_per_fact=args.n_per_fact)
    H, rel_to_facts = build_relation_hypergraph(facts, args.n_per_fact)
    cycles = find_fundamental_cycles(H.nodes, H.edges)

    model, tokenizer = load_quantized(args.model, args.quantize)
    device = next(model.parameters()).device

    node_features: Dict[str, torch.Tensor] = {}
    for i, fact in enumerate(facts):
        for j in range(min(args.n_per_fact, len(fact['paraphrases']))):
            name = f'f{i}_p{j}'
            node_features[name] = extract_token_activations(
                model, tokenizer, fact['paraphrases'][j],
                args.layer, device, args.n_token_samples,
            )
    del model
    torch.cuda.empty_cache()

    g = torch.Generator(device=device).manual_seed(args.seed)
    proj = fit_per_node_pca(
        {n: f.to(device) for n, f in node_features.items()},
        args.edge_dim, device, g,
    )
    transports = fit_procrustes_transports(
        node_features, proj, H.edges, device,
    )
    obs_obstr = cycle_obstructions(
        cycles, transports, args.edge_dim, device,
    ).numpy()

    shuffle_means: List[float] = []
    shuffle_stds: List[float] = []
    for s in range(args.n_shuffle_seeds):
        shuffled = shuffle_paraphrases_within_relation(
            node_features, rel_to_facts, args.n_per_fact,
            seed=args.seed + 1000 + s,
        )
        gs = torch.Generator(device=device).manual_seed(args.seed + 1000 + s)
        sproj = fit_per_node_pca(
            {n: f.to(device) for n, f in shuffled.items()},
            args.edge_dim, device, gs,
        )
        st = fit_procrustes_transports(shuffled, sproj, H.edges, device)
        sobstr = cycle_obstructions(
            cycles, st, args.edge_dim, device,
        ).numpy()
        shuffle_means.append(float(sobstr.mean()))
        shuffle_stds.append(float(sobstr.std()))
    shuffle_arr = np.array(shuffle_means)
    p_value = float(
        ((shuffle_arr >= obs_obstr.mean()).sum() + 1)
        / (len(shuffle_arr) + 1)
    )

    bootstrap: Dict[int, Dict[str, float]] = {}
    for N in args.bootstrap_n_values:
        if N > args.n_token_samples:
            continue
        truncated = {n: f[:N] for n, f in node_features.items()}
        gb = torch.Generator(device=device).manual_seed(args.seed + N)
        bp = fit_per_node_pca(
            {n: f.to(device) for n, f in truncated.items()},
            args.edge_dim, device, gb,
        )
        bt = fit_procrustes_transports(
            truncated, bp, H.edges, device, n_overlap_tokens=N,
        )
        bobs = cycle_obstructions(
            cycles, bt, args.edge_dim, device,
        ).numpy()
        bootstrap[N] = {
            'obstruction_mean': float(bobs.mean()),
            'obstruction_std': float(bobs.std()),
        }

    eye = torch.eye(args.edge_dim, device=device)
    id_transports = {e: eye.clone() for e in H.edges}
    id_obstr = cycle_obstructions(
        cycles, id_transports, args.edge_dim, device,
    ).numpy()

    payload = {
        'model': args.model,
        'layer': args.layer,
        'edge_dim': args.edge_dim,
        'n_per_fact': args.n_per_fact,
        'n_token_samples': args.n_token_samples,
        'n_nodes': len(H.nodes),
        'n_edges': len(H.edges),
        'n_cycles': len(cycles),
        'frobenius_upper_bound_sqrt_2k': float(np.sqrt(2 * args.edge_dim)),
        'observed': {
            'obstruction_mean': float(obs_obstr.mean()),
            'obstruction_std': float(obs_obstr.std()),
            'obstruction_per_cycle': obs_obstr.tolist(),
        },
        'shuffled_null': {
            'n_shuffle_seeds': args.n_shuffle_seeds,
            'obstruction_means_per_seed': shuffle_means,
            'mean_of_means': float(shuffle_arr.mean()),
            'sd_of_means': float(shuffle_arr.std()),
            'one_sided_p_shuffled_ge_observed': p_value,
        },
        'bootstrap_N': bootstrap,
        'identity_null': {
            'obstruction_mean': float(id_obstr.mean()),
            'obstruction_std': float(id_obstr.std()),
        },
        'seed': args.seed,
        'timestamp': ts,
    }
    out_path.write_text(json.dumps(payload, indent=2))


if __name__ == '__main__':
    main()
