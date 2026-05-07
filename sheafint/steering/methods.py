"""
Build steering vectors from a sheaf decomposition.

Takes a target style direction (e.g. casual minus formal) plus a sheaf
or Fisher decomposition and produces the variance-matched, H^1-only,
H^0-removed, and full-style steering vectors compared in Table 4.
"""

from typing import Dict
import numpy as np
import torch


def build_steering_vectors(
    style_vector: np.ndarray,
    decomposition: Dict,
    rng: np.random.Generator,
    mod_vector: np.ndarray = None,
) -> Dict[str, np.ndarray]:
    # Construct random / full-style / variance / H1-dim / H0-removed steering vectors.
    d = style_vector.shape[0]

    random_vec = rng.standard_normal(d).astype(np.float64)
    random_vec /= (np.linalg.norm(random_vec) + 1e-10)
    random_vec *= np.linalg.norm(style_vector)

    methods = {
        'random': random_vec,
        'full_style': style_vector.copy(),
        'mod': mod_vector.copy() if mod_vector is not None else style_vector.copy(),
    }

    variance_proj = np.zeros(d)
    variance_proj[decomposition['var_h1']] = style_vector[decomposition['var_h1']]
    methods['variance'] = variance_proj

    h1_proj = np.zeros(d)
    h1_proj[decomposition['h1_dims']] = style_vector[decomposition['h1_dims']]
    methods['h1_dims'] = h1_proj

    h0_basis = decomposition['h0_vecs']
    methods['h0_removed'] = style_vector - h0_basis @ (h0_basis.T @ style_vector)

    return methods


def scale_to_norm(vectors: Dict[str, np.ndarray], target_norm: float) -> Dict[str, torch.Tensor]:
    # Rescale every steering vector to a shared target norm so methods are comparable.
    scaled: Dict[str, torch.Tensor] = {}
    for name, vec in vectors.items():
        norm = np.linalg.norm(vec)
        if norm > 1e-8:
            scaled[name] = torch.tensor(vec * (target_norm / norm), dtype=torch.float32)
        else:
            scaled[name] = torch.tensor(vec, dtype=torch.float32)
    return scaled
