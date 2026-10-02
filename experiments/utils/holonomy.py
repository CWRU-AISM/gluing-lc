"""
Holonomy null calibration helpers: last-token activations per paraphrase,
within-relation paraphrase shuffles, and the per-cycle obstruction
||I - H_gamma||_F of per-node PCA + Procrustes transports.
"""

from typing import Dict, List

import numpy as np
import torch

from sheafint import cycle_obstructions, fit_per_node_pca, fit_procrustes_transports


@torch.no_grad()
def token_activations(model, tokenizer, prompt: str, layer: int, device, n_last: int = 6) -> torch.Tensor:
    """Float32 hidden states of the last ``n_last`` tokens of one prompt at ``hidden_states[layer]``."""
    ids = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=64).to(device)
    h = model(**ids, output_hidden_states=True).hidden_states[layer].squeeze(0)
    return h[-min(n_last, h.shape[0]):].float().cpu()


def shuffle_within_relation(
    node_features: Dict[str, torch.Tensor],
    rel_to_facts: Dict[str, List[int]],
    n_per_fact: int,
    seed: int,
) -> Dict[str, torch.Tensor]:
    """Permute which paraphrase node holds which features among the facts of each relation."""
    rng = np.random.default_rng(seed)
    shuffled: Dict[str, torch.Tensor] = {}
    for fact_ids in rel_to_facts.values():
        names = [f'f{fid}_p{p}' for fid in fact_ids for p in range(n_per_fact) if f'f{fid}_p{p}' in node_features]
        perm = rng.permutation(len(names))
        for j, name in enumerate(names):
            shuffled[name] = node_features[names[perm[j]]]
    for name, feat in node_features.items():
        shuffled.setdefault(name, feat)
    return shuffled


def holonomy_obstructions(node_features, edges, cycles, edge_dim: int, device, seed: int,
                          n_overlap_tokens: int = None) -> np.ndarray:
    """Per-cycle ||I - H_gamma||_F after fitting per-node PCA (seeded padding) and Procrustes transports."""
    g = torch.Generator(device=device).manual_seed(seed)
    proj = fit_per_node_pca({n: f.to(device) for n, f in node_features.items()}, edge_dim, device, g)
    transports = fit_procrustes_transports(node_features, proj, edges, device, n_overlap_tokens=n_overlap_tokens)
    return cycle_obstructions(cycles, transports, edge_dim, device).numpy()


def mean_std(x: np.ndarray) -> Dict[str, float]:
    return {'obstruction_mean': float(x.mean()), 'obstruction_std': float(x.std())}
