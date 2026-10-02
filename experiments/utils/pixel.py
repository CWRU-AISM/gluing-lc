"""
PIXEL (Yu et al., 2025, arXiv:2510.10205) adapted to free-form CounterFact generation.

1. Dual-view direction: per casual/formal exemplar pair, the difference of the
   mean of the last k_tail = clip(0.1 * prompt_len, 3, 8) prompt-token states
   and of the end-token states at the steered layer; the top weighted-PCA
   component of all differences, sign-aligned with their mean.
2. Closed-form magnitude per prompt position t in the last k_tail tokens: with
   a = <h_t, u>, B = sqrt(|h_t|^2 - a^2) and target z-score s,
   alpha_t = clip(max(-a, B s / sqrt(1 - s^2) - a), alpha_clip).
3. alpha_t * u is added at those positions on the prompt forward pass only.
"""

import math

import numpy as np
import torch

from .model_io import generate, get_layers
from .style_examples import CASUAL_EXAMPLES, FORMAL_EXAMPLES


def tail_length(prompt_len: int) -> int:
    return int(np.clip(round(0.10 * max(1, prompt_len)), 3, 8))


@torch.inference_mode()
def prompt_states(model, tokenizer, text: str, layer: int, max_length=None):
    """Per-token hidden states at the output of block ``layer`` for one unpadded prompt."""
    device = next(model.parameters()).device
    enc = tokenizer(text, return_tensors='pt', truncation=max_length is not None, max_length=max_length).to(device)
    return model(**enc, output_hidden_states=True).hidden_states[layer + 1][0], enc


def views(model, tokenizer, text: str, layer: int):
    """(tail-mean, end-token) float32 states of one prompt."""
    hs, enc = prompt_states(model, tokenizer, text, layer, max_length=128)
    plen = int(enc['attention_mask'].sum().item())
    tail = hs[plen - tail_length(plen):plen].float().cpu().numpy().mean(axis=0)
    return tail, hs[plen - 1].float().cpu().numpy()


def weighted_pca_direction(X: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Unit top principal direction of the column-standardised rows of ``X``."""
    X = X.astype(np.float32)
    Xc = X - X.mean(0, keepdims=True)
    v = np.linalg.svd(Xc / (Xc.std(0, keepdims=True) + eps), full_matrices=False)[2][0].astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-9)


def pixel_direction(model, tokenizer, layer: int) -> np.ndarray:
    """Dual-view casual-minus-formal direction, sign-aligned with the mean difference."""
    tail_diffs, end_diffs = [], []
    for casual, formal in zip(CASUAL_EXAMPLES, FORMAL_EXAMPLES):
        (ct, ce), (ft, fe) = views(model, tokenizer, casual, layer), views(model, tokenizer, formal, layer)
        tail_diffs.append((ct - ft).astype(np.float32))
        end_diffs.append((ce - fe).astype(np.float32))
    X = np.concatenate([np.stack(tail_diffs), np.stack(end_diffs)], axis=0)
    u = weighted_pca_direction(X)
    return -u if np.dot(u, X.mean(axis=0)) < 0 else u


def closed_form_alphas(H: torch.Tensor, u: torch.Tensor, positions, z_target: float, alpha_clip, alpha_max=None):
    """Per-position magnitudes that lift the cosine with the float32 direction ``u`` to ``z_target``."""
    u_dev = u.to(H.device).to(H.dtype)
    c = float(u.float().norm().item()) + 1e-9
    s = max(1e-6, min(1.0 - 1e-6, float(z_target)))
    lo, hi = alpha_clip
    alphas = {}
    for t in positions:
        h = H[t].to(u_dev.dtype)
        a = float(torch.dot(h, u_dev).item())
        hn = float(h.float().norm().item()) + 1e-9
        target = math.sqrt(max(hn * hn - a * a, 0.0)) * s / math.sqrt(max(1.0 - s * s, 1e-9))
        alpha = max(lo, min(hi, max(max(0.0, -a / max(c, 1e-9)), (target - a) / max(c, 1e-9))))
        if alpha_max is not None:
            alpha = min(alpha, alpha_max)
        if alpha > 0.0:
            alphas[t] = alpha
    return alphas


def generate_with_pixel(model, tokenizer, prompt, layer, u, max_new_tokens, z_target, alpha_clip, alpha_max=None):
    """Greedy generation with PIXEL injection at the last k_tail prompt positions."""
    H, enc = prompt_states(model, tokenizer, prompt, layer)
    plen = int(enc['attention_mask'].sum().item())
    alphas = closed_form_alphas(H, u, range(max(0, plen - tail_length(plen)), plen), z_target, alpha_clip, alpha_max)
    u_dev = u.to(H.device).to(H.dtype)
    if not alphas:
        return generate(model, tokenizer, prompt, max_new_tokens)

    def hook(_module, _inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        if h is None or h.shape[1] < plen:  # generation steps see only new tokens
            return output
        mask = torch.zeros((1, h.shape[1], 1), dtype=h.dtype, device=h.device)
        for t, a in alphas.items():
            mask[0, t, 0] = a
        h = h + mask * u_dev.to(h.dtype).view(1, 1, -1)
        return (h,) + output[1:] if isinstance(output, tuple) else h

    handle = get_layers(model)[layer].register_forward_hook(hook)
    try:
        return generate(model, tokenizer, prompt, max_new_tokens)
    finally:
        handle.remove()


@torch.inference_mode()
def mean_pooled_norm(model, tokenizer, texts, layer, batch_size=16) -> float:
    """Mean norm of the mean-pooled block-``layer`` outputs of ``texts``."""
    device = next(model.parameters()).device
    norms = []
    for i in range(0, len(texts), batch_size):
        enc = tokenizer(texts[i:i + batch_size], return_tensors='pt', padding=True, truncation=True, max_length=128)
        enc = {k: v.to(device) for k, v in enc.items()}
        hs = model(**enc, output_hidden_states=True).hidden_states[layer + 1]
        mask = enc['attention_mask'].unsqueeze(-1)
        norms.append(((hs * mask).sum(1) / mask.sum(1)).norm(dim=-1).float().cpu())
    return float(torch.cat(norms).mean())


def reference_vectors(model, tokenizer, layer, target_norm, seed):
    """Norm-matched random and end-token CAA vectors at ``target_norm``."""
    d = model.config.hidden_size
    rand = np.random.default_rng(seed).standard_normal(d).astype(np.float32)
    rand /= (np.linalg.norm(rand) + 1e-9)
    casual = np.stack([views(model, tokenizer, t, layer)[1] for t in CASUAL_EXAMPLES]).mean(0)
    formal = np.stack([views(model, tokenizer, t, layer)[1] for t in FORMAL_EXAMPLES]).mean(0)
    style = torch.from_numpy((casual - formal).astype(np.float32))
    return {
        'random': torch.from_numpy(rand * target_norm),
        'full_style': style * (target_norm / (style.norm() + 1e-9)),
    }
