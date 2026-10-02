"""
Steering helpers built on top of sheaf decompositions.

Re-exports the restriction-map fitters (joint PCA, CCA, contrastive),
the sheaf-Laplacian and Fisher decompositions of the projected residuals,
and the steering-vector builders used by the CounterFact experiments.
"""

from .decomposition import fisher_decomposition, sheaf_laplacian_decomposition
from .methods import baseline_vectors, build_steering_vectors, decomposition_vectors, scale_to_norm
from .restriction_maps import cca, contrastive, joint_pca

__all__ = [
    'joint_pca', 'cca', 'contrastive',
    'sheaf_laplacian_decomposition', 'fisher_decomposition',
    'baseline_vectors', 'decomposition_vectors', 'build_steering_vectors', 'scale_to_norm',
]
