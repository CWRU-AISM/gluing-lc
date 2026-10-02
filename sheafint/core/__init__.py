"""
Core sheaf primitives.

Re-exports :class:`ScalableSheaf` along with the H^0 fitters, edge
constructors, and holonomy helpers used by the experiment scripts.
"""

from .h0 import (
    fit_h0_from_edges,
    fit_h0_identity,
    fit_h0_pca,
    fit_h0_random,
    fit_h0_with_projection,
)
from .graphs import (
    clique_edges,
    count_clique_cycles,
    count_ring_cycles,
    pair_edges,
    ring_edges,
)
from .holonomy import (
    cycle_holonomy,
    cycle_obstructions,
    find_fundamental_cycles,
    fit_per_node_pca,
    fit_procrustes_transports,
)
from .sheaf import ScalableSheaf

__all__ = [
    'ScalableSheaf',
    'fit_h0_with_projection',
    'fit_h0_pca',
    'fit_h0_random',
    'fit_h0_identity',
    'fit_h0_from_edges',
    'pair_edges',
    'ring_edges',
    'clique_edges',
    'count_ring_cycles',
    'count_clique_cycles',
    'cycle_holonomy',
    'cycle_obstructions',
    'find_fundamental_cycles',
    'fit_per_node_pca',
    'fit_procrustes_transports',
]
