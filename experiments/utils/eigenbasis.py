"""
Sheaf eigenbasis ablation: block-output pooling, activation energy, and the
KL-ratio summaries against Haar, span(P) and PCA control subspaces.
"""

import numpy as np
import torch
from scipy.stats import wilcoxon

from .model_io import get_layers
from .statistics import bootstrap_ci

EPS = 1e-12


def default_layers(n: int):
    """Every other block, always including the last."""
    layers = list(range(1, n, 2))
    return layers if n - 1 in layers else layers + [n - 1]


@torch.no_grad()
def pooled_block_outputs(model, tokenizer, prompts, layers, batch_size, max_length=64):
    """Mean-pooled float32 output of each block in ``layers`` (the ablation hook site).

    Hooks the blocks directly: hidden_states[-1] is post-final-norm, so
    hidden_states[l + 1] would not match the hook site at the last layer.
    """
    device = next(model.parameters()).device
    blocks = get_layers(model)
    grabbed, out = {}, {l: [] for l in layers}
    handles = [blocks[l].register_forward_hook(
        lambda m, i, o, l=l: grabbed.__setitem__(l, o[0] if isinstance(o, tuple) else o))
        for l in layers]
    try:
        for i in range(0, len(prompts), batch_size):
            enc = tokenizer(prompts[i:i + batch_size], return_tensors='pt', padding=True,
                            truncation=True, max_length=max_length).to(device)
            model(**enc)
            m = enc['attention_mask'].unsqueeze(-1).float()
            for l in layers:
                out[l].append(((grabbed[l].float() * m).sum(1) / m.sum(1).clamp(min=1)).cpu())
    finally:
        for h in handles:
            h.remove()
    return {l: torch.cat(v) for l, v in out.items()}


def energy(U, S):
    """Fraction of the second moment tr(S) that projecting out span(U) removes."""
    return float(np.trace(U.T @ S @ U) / np.trace(S))


def _paired_ratio_ci(kl_h1, ctrl, n_boot, seed):
    idx = np.random.default_rng(seed).integers(0, len(kl_h1), (n_boot, len(kl_h1)))
    boot = kl_h1[idx].mean(1) / ctrl[idx].mean(1)
    return [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]


def versus(kl_h1, kl_ctrl, n_boot, seed):
    """H^1 against one control family; ``kl_ctrl`` is (n_draws, n)."""
    ctrl = kl_ctrl.mean(0)
    return {'ratio_of_means': float(kl_h1.mean() / ctrl.mean()),
            'ratio_of_means_ci': _paired_ratio_ci(kl_h1, ctrl, n_boot, seed),
            'median_ratio': float(np.median(kl_h1 / np.maximum(ctrl, EPS))),
            'frac_h1_above_max': float((kl_h1 > kl_ctrl.max(0)).mean()),
            'wilcoxon_p': float(wilcoxon(kl_h1, ctrl).pvalue)}


def summarize(kl_h1, kl_h0, kl_rand, n_boot, seed):
    """H^1 and H^0 against the Haar controls; ``kl_rand`` is (n_controls, n)."""
    rand_mean = kl_rand.mean(0)
    ratio = kl_h1 / np.maximum(rand_mean, EPS)
    _, r_lo, r_hi = bootstrap_ci(ratio, n_boot=n_boot, seed=seed)
    dlog = np.log(np.maximum(kl_h1, EPS)) - np.log(np.maximum(rand_mean, EPS))
    return {
        'ratio_of_means': float(kl_h1.mean() / rand_mean.mean()),
        'ratio_of_means_ci': _paired_ratio_ci(kl_h1, rand_mean, n_boot, seed),
        'median_ratio': float(np.median(ratio)),
        'mean_ratio': float(ratio.mean()),
        'mean_ratio_ci': [r_lo, r_hi],
        'frac_h1_above_max_rand': float((kl_h1 > kl_rand.max(0)).mean()),
        'wilcoxon_p': float(wilcoxon(kl_h1, rand_mean).pvalue),
        'cohens_d_logkl': float(dlog.mean() / dlog.std(ddof=1)),
        'h0_ratio_of_means': float(kl_h0.mean() / rand_mean.mean()),
        'h0_median_ratio': float(np.median(kl_h0 / np.maximum(rand_mean, EPS))),
    }
