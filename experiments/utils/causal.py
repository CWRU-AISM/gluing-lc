"""
Helpers for ablation-based causal validation of H0 / H1 dimensions.

Provides a hook factory that zeros chosen residual-stream dimensions, an
end-to-end effect estimator that compares baseline and ablated final-layer
states, the variance-proxy H0/H1 dim identifier, the variance-matched
control-dim selector, and the bootstrap CI used in Table 1.
"""

from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .model_io import get_layers


def device_for_inputs(model) -> str:
    """
    Pick a sensible device for inputs even when the model is sharded.

    Prefers the first CUDA shard recorded in ``hf_device_map`` and falls
    back to the device of the first parameter tensor.
    """
    if hasattr(model, 'hf_device_map'):
        for dev in model.hf_device_map.values():
            if 'cuda' in str(dev):
                return dev
    return next(model.parameters()).device


@torch.no_grad()
def hidden_states_at(model, tokenizer, text: str, layer_idx: int) -> torch.Tensor:
    """
    Hidden states at the post-block of layer ``layer_idx`` for one input.

    Uses ``layer_idx + 1`` against ``output_hidden_states`` because index 0
    is the embedding layer.
    """
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=128)
    inputs = inputs.to(device_for_inputs(model))
    outputs = model(inputs['input_ids'], output_hidden_states=True, return_dict=True)
    return outputs.hidden_states[layer_idx + 1][0].to(device_for_inputs(model))


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


def ablation_effect(
    model,
    tokenizer,
    text: str,
    layer_idx: int,
    n_layers_total: int,
    dims: List[int],
) -> Dict[str, float]:
    """
    L2 and cosine distance between baseline and ablated final-layer states.

    Substitutes the baseline state in place of any NaN/Inf values from the
    ablated forward pass so a single bad batch cannot crash the aggregator.
    """
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
) -> Tuple[List[int], List[int], np.ndarray]:
    """
    Variance-proxy H0 / H1 dimension picker.

    Computes the mean per-dim absolute difference across paraphrase pairs;
    H0 dims are the lowest-difference (most invariant) and H1 dims are the
    highest-difference (most variable).
    """
    diffs = []
    for s1, s2 in paraphrase_pairs:
        try:
            h_a = hidden_states_at(model, tokenizer, s1, layer_idx).mean(dim=0)
            h_b = hidden_states_at(model, tokenizer, s2, layer_idx).mean(dim=0)
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
    available.sort(key=lambda d: abs(variance_per_dim[d] - target_each))
    return available[:n_dims]


def bootstrap_effect_ratio(
    condition_a: List[float],
    condition_b: List[float],
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> Dict:
    """
    Bootstrap CI for the ``mean(a) / mean(b)`` ratio.

    Returns the point ratio, 2.5 / 97.5 percentile bounds, and a
    significance flag set when the 95% CI excludes 1.
    """
    rng = np.random.default_rng(seed)
    a = np.asarray(condition_a)
    b = np.asarray(condition_b)
    ratio = a.mean() / (b.mean() + 1e-8)
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
