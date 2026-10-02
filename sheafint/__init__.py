"""
sheafint: cellular sheaf cohomology for neural-network interpretability.

Public surface re-exports the most-used helpers from :mod:`sheafint.core`
so user scripts can write ``from sheafint import fit_h0_pca`` without
knowing the package layout.
"""

from .core import (
    ScalableSheaf,
    clique_edges,
    count_clique_cycles,
    count_ring_cycles,
    cycle_holonomy,
    cycle_obstructions,
    find_fundamental_cycles,
    fit_h0_from_edges,
    fit_h0_identity,
    fit_h0_pca,
    fit_h0_random,
    fit_h0_with_projection,
    fit_per_node_pca,
    fit_procrustes_transports,
    pair_edges,
    ring_edges,
)

__version__ = '0.1.0'
__all__ = [
    'ScalableSheaf',
    'fit_h0_with_projection', 'fit_h0_pca', 'fit_h0_random', 'fit_h0_identity',
    'fit_h0_from_edges',
    'pair_edges', 'ring_edges', 'clique_edges',
    'count_ring_cycles', 'count_clique_cycles',
    'cycle_holonomy', 'cycle_obstructions', 'find_fundamental_cycles',
    'fit_per_node_pca', 'fit_procrustes_transports',
]
