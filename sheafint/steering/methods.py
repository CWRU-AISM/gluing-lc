"""
Build steering vectors from a sheaf decomposition.

Takes a target style direction (e.g. casual minus formal) plus a sheaf
or Fisher decomposition and produces the variance-matched, H^1-only,
H^0-removed, and full-style steering vectors compared in Table 4.
"""

from typing import Dict, Sequence

import numpy as np
import torch


def baseline_vectors(
    style_vector: np.ndarray,
    var_dims: Sequence[int],
    rng: np.random.Generator,
    mod_vector: np.ndarray = None,
) -> Dict[str, np.ndarray]:
    """Random (style-norm), full-style (CAA), MoD and variance top-k steering vectors."""
    random_vec = rng.standard_normal(style_vector.shape[0]).astype(np.float64)
    random_vec /= (np.linalg.norm(random_vec) + 1e-10)
    random_vec *= np.linalg.norm(style_vector)

    variance_proj = np.zeros(style_vector.shape[0])
    variance_proj[var_dims] = style_vector[var_dims]
    return {
        'random': random_vec,
        'full_style': style_vector.copy(),
        'mod': mod_vector.copy() if mod_vector is not None else style_vector.copy(),
        'variance': variance_proj,
    }


def decomposition_vectors(style_vector: np.ndarray, decomposition: Dict) -> Dict[str, np.ndarray]:
    """The style vector restricted to the H^1 coordinates, and with its H^0 component removed."""
    h1_proj = np.zeros(style_vector.shape[0])
    h1_proj[decomposition['h1_dims']] = style_vector[decomposition['h1_dims']]
    h0_basis = decomposition['h0_vecs']
    return {
        'h1_dims': h1_proj,
        'h0_removed': style_vector - h0_basis @ (h0_basis.T @ style_vector),
    }


def build_steering_vectors(
    style_vector: np.ndarray,
    decomposition: Dict,
    rng: np.random.Generator,
    mod_vector: np.ndarray = None,
) -> Dict[str, np.ndarray]:
    """Random / full-style / MoD / variance / H1-dim / H0-removed steering vectors."""
    return {
        **baseline_vectors(style_vector, decomposition['var_h1'], rng, mod_vector),
        **decomposition_vectors(style_vector, decomposition),
    }


def scale_to_norm(vectors: Dict[str, np.ndarray], target_norm: float) -> Dict[str, torch.Tensor]:
    """Rescale every steering vector to a shared target norm so methods are comparable."""
    scaled: Dict[str, torch.Tensor] = {}
    for name, vec in vectors.items():
        norm = np.linalg.norm(vec)
        if norm > 1e-8:
            scaled[name] = torch.tensor(vec * (target_norm / norm), dtype=torch.float32)
        else:
            scaled[name] = torch.tensor(vec, dtype=torch.float32)
    return scaled
