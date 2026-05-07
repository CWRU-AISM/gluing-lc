"""
sheafint: cellular sheaf cohomology for neural-network interpretability.

Public surface re-exports the most-used helpers from :mod:`sheafint.core`,
:mod:`sheafint.data`, and :mod:`sheafint.models` so user scripts can write
``from sheafint import fit_h0_pca`` without knowing the package layout.
"""

from .core import (
    ScalableSheaf,
    SheafMetrics,
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
from .data import (
    BaseContext, ContextCover, ContextPair,
    MRPCContext, PAWSContext, ParaphraseContext, PromptVariantContext,
    QQPContext, STSBContext, TranslationContext,
    create_full_cover, create_mrpc_cover, create_standard_cover,
)
from .models import (
    ActivationCache, HookManager,
    create_extraction_hooks, extract_features, extract_features_for_contexts,
)

__version__ = '0.1.0'
__all__ = [
    'ScalableSheaf', 'SheafMetrics',
    'fit_h0_with_projection', 'fit_h0_pca', 'fit_h0_random', 'fit_h0_identity',
    'fit_h0_from_edges',
    'pair_edges', 'ring_edges', 'clique_edges',
    'count_ring_cycles', 'count_clique_cycles',
    'cycle_holonomy', 'cycle_obstructions', 'find_fundamental_cycles',
    'fit_per_node_pca', 'fit_procrustes_transports',
    'BaseContext', 'ContextPair', 'ContextCover',
    'ParaphraseContext', 'MRPCContext', 'QQPContext', 'STSBContext', 'PAWSContext',
    'TranslationContext', 'PromptVariantContext',
    'create_standard_cover', 'create_mrpc_cover', 'create_full_cover',
    'ActivationCache', 'HookManager',
    'create_extraction_hooks', 'extract_features', 'extract_features_for_contexts',
]
