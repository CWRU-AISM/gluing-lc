"""
Helpers for subspace and coordinate ablations.

Subspace ablation (eigenbasis and dormancy experiments): Haar-random control
subspaces, a forward hook that projects the residual stream onto the
orthogonal complement of a subspace, the projection spread used as the
dormancy diagnostic, greedy generation, next-token logits at the final
prompt position, and the next-token KL used as the ablation effect.

Coordinate ablation (causal validation table): variance-proxy H^0 / H^1
coordinate selection, variance-matched controls, the ablation effect on the
final hidden state or logits, and a bootstrap CI on the effect ratio.
Protocol (a) of ``causal.py coordinates`` ranks coordinates and measures
logits in the model dtype; protocol (b) works in float32.
"""

from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .model_io import device_for_inputs, get_layers, hidden_states_at


def haar_basis(d: int, k: int, rng) -> np.ndarray:
    """Haar-uniform random orthonormal (d, k) basis (QR of a Gaussian matrix)."""
    Q, _ = np.linalg.qr(rng.standard_normal((d, k)))
    return Q


class ProjectionAblator:
    """Forward hook: h <- h - (h U) U^T at every position of the layer output."""

    def __init__(self, U, device, dtype):
        self.U = torch.from_numpy(U).to(device=device, dtype=dtype)
        self.enabled = False

    def __call__(self, module, inputs, output):
        if not self.enabled:
            return output
        hidden = output[0] if isinstance(output, tuple) else output
        proj = (hidden.to(self.U.dtype) @ self.U) @ self.U.T
        hidden = hidden - proj.to(hidden.dtype)
        if isinstance(output, tuple):
            return (hidden,) + tuple(output[1:])
        return hidden


def projection_spread(H: np.ndarray, U: np.ndarray) -> float:
    """Std of the projections of H (n, d) onto each column of U (d, k), averaged."""
    return float((H @ U).std(axis=0).mean())


@torch.no_grad()
def greedy_generation(model, tokenizer, prompt: str, device, max_new: int = 50) -> str:
    """Greedy continuation of ``prompt``, decoded without the prompt tokens."""
    enc = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=64).to(device)
    gen = model.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tokenizer.pad_token_id)
    return tokenizer.decode(gen[0, enc['input_ids'].shape[1]:], skip_special_tokens=True)


@torch.no_grad()
def last_token_logits(model, tokenizer, prompts, device, max_length: int = 64):
    """
    Next-token logits at the final non-pad position.

    A single string gives ``(V,)``; a list gives ``(B, V)`` and assumes
    right padding (``tokenizer.padding_side = 'right'``).
    """
    enc = tokenizer(prompts, return_tensors='pt', padding=True,
                    truncation=True, max_length=max_length).to(device)
    logits = model(**enc).logits
    last = (enc['attention_mask'].sum(1) - 1).to(logits.device)
    out = logits[torch.arange(len(last), device=logits.device), last].float()
    return out[0] if isinstance(prompts, str) else out


def kl_divergence(logits_p, logits_q):
    """KL(p || q) over the last dim; a float for 1-D input, a list for batched."""
    p = F.log_softmax(logits_p.double(), dim=-1)
    q = F.log_softmax(logits_q.double(), dim=-1)
    return (p.exp() * (p - q)).sum(-1).tolist()


def make_ablation_hook(intervention_dims: List[int]):
    """
    Build a forward hook that zeros the requested residual-stream dims.

    Out-of-range dims are silently dropped so callers can pass dim lists
    that exceed the model's hidden size without manual filtering.
    """
    def hook(_module, _inputs, output):
        if not intervention_dims:
            return output
        if isinstance(output, tuple):
            hidden = output[0].clone()
        else:
            hidden = output.clone()
        max_dim = hidden.shape[-1]
        valid = [d for d in intervention_dims if 0 <= d < max_dim]
        if valid:
            hidden[:, :, valid] = 0
        if isinstance(output, tuple):
            return (hidden,) + output[1:]
        return hidden
    return hook


@torch.no_grad()
def last_position_logits(model, tokenizer, text: str) -> torch.Tensor:
    """Final-position logits for one unpadded input, in the model dtype (protocol a)."""
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=128)
    return model(inputs['input_ids'].to(device_for_inputs(model)), return_dict=True).logits[0, -1]


def ablation_effect(
    model,
    tokenizer,
    text: str,
    layer_idx: int,
    n_layers_total: int,
    dims: List[int],
    effect: str = 'hidden',
) -> Dict[str, float]:
    """
    L2 and cosine distance between baseline and ablated final-layer states.

    ``effect='logits'`` instead returns the L2 change of the final-position
    logits, computed in the model dtype. Otherwise substitutes the baseline
    state in place of any NaN/Inf values from the ablated forward pass so a
    single bad batch cannot crash the aggregator.
    """
    if effect == 'logits':
        baseline = last_position_logits(model, tokenizer, text)
        handle = get_layers(model)[layer_idx].register_forward_hook(make_ablation_hook(dims))
        try:
            ablated = last_position_logits(model, tokenizer, text)
        finally:
            handle.remove()
        return {'l2_distance': (baseline - ablated).norm().item()}

    baseline = hidden_states_at(model, tokenizer, text, n_layers_total - 1).mean(dim=0)
    handle = get_layers(model)[layer_idx].register_forward_hook(make_ablation_hook(dims))
    try:
        ablated = hidden_states_at(model, tokenizer, text, n_layers_total - 1).mean(dim=0)
    finally:
        handle.remove()

    baseline_cpu = baseline.cpu().float()
    ablated_cpu = ablated.cpu().float()
    if torch.isnan(ablated_cpu).any():
        ablated_cpu = torch.where(torch.isnan(ablated_cpu), baseline_cpu, ablated_cpu)
    if torch.isinf(ablated_cpu).any():
        ablated_cpu = torch.where(torch.isinf(ablated_cpu), baseline_cpu, ablated_cpu)

    return {
        'l2_distance': torch.norm(ablated_cpu - baseline_cpu).item(),
        'cosine_similarity': F.cosine_similarity(
            baseline_cpu.unsqueeze(0), ablated_cpu.unsqueeze(0)
        ).item(),
    }


def identify_h0_h1_dims(
    model,
    tokenizer,
    paraphrase_pairs: List[Tuple[str, str]],
    layer_idx: int,
    n_dims: int,
    model_dtype_diffs: bool = False,
) -> Tuple[List[int], List[int], np.ndarray]:
    """
    Variance-proxy H0 / H1 dimension picker.

    Computes the mean per-dim absolute difference across paraphrase pairs;
    H0 dims are the lowest-difference (most invariant) and H1 dims are the
    highest-difference (most variable). ``model_dtype_diffs=True`` takes the
    difference and its mean in the model dtype (fp16, protocol a); fp16 ties
    then decide the control set.
    """
    diffs = []
    for s1, s2 in paraphrase_pairs:
        try:
            h_a = hidden_states_at(model, tokenizer, s1, layer_idx).mean(dim=0)
            h_b = hidden_states_at(model, tokenizer, s2, layer_idx).mean(dim=0)
            if model_dtype_diffs:
                diffs.append((h_a - h_b).abs().cpu().numpy())
            else:
                diffs.append((h_a.cpu().float() - h_b.cpu().float()).abs().numpy())
        except Exception:
            continue

    if not diffs:
        return [], [], np.array([])

    variance_per_dim = np.stack(diffs).mean(axis=0)
    sorted_idx = np.argsort(variance_per_dim)
    h0_dims = sorted_idx[:n_dims].tolist()
    h1_dims = sorted_idx[-n_dims:][::-1].tolist()
    return h0_dims, h1_dims, variance_per_dim


def variance_matched_dims(
    target_variance: float,
    variance_per_dim: np.ndarray,
    n_dims: int,
    exclude_dims: List[int],
) -> List[int]:
    """
    Pick dims whose total variance roughly matches ``target_variance``.

    Sorted by per-dim distance to the equal-share target so the chosen
    subset is variance-matched without overlapping the H0 / H1 sets passed
    in via ``exclude_dims``.
    """
    available = [i for i in range(len(variance_per_dim)) if i not in exclude_dims]
    target_each = target_variance / n_dims
    if variance_per_dim.dtype == np.float16:  # protocol (a): fp16 distances, ties broken by index
        target_each = np.float16(target_each)
    available.sort(key=lambda d: abs(variance_per_dim[d] - target_each))
    return available[:n_dims]


def bootstrap_effect_ratio(
    condition_a: List[float],
    condition_b: List[float],
    n_bootstrap: int = 1000,
    seed: int = 42,
    paired: bool = False,
) -> Dict:
    """
    Bootstrap CI for the ``mean(a) / mean(b)`` ratio.

    Returns the point ratio, 2.5 / 97.5 percentile bounds, and a
    significance flag set when the 95% CI excludes 1. ``paired=True``
    resamples one index set for both conditions (protocol a).
    """
    rng = np.random.default_rng(seed)
    a = np.asarray(condition_a)
    b = np.asarray(condition_b)
    ratio = a.mean() / (b.mean() + 1e-8)
    if paired:
        idx = [rng.integers(0, len(a), size=len(a)) for _ in range(n_bootstrap)]
        ratios = np.array([a[i].mean() / (b[i].mean() + 1e-8) for i in idx])
    else:
        ratios = np.array([
            a[rng.integers(0, len(a), size=len(a))].mean()
            / (b[rng.integers(0, len(b), size=len(b))].mean() + 1e-8)
            for _ in range(n_bootstrap)
        ])
    return {
        'ratio': float(ratio),
        'ci_lower': float(np.percentile(ratios, 2.5)),
        'ci_upper': float(np.percentile(ratios, 97.5)),
        'significant': bool(
            np.percentile(ratios, 2.5) > 1.0 or np.percentile(ratios, 97.5) < 1.0
        ),
    }
