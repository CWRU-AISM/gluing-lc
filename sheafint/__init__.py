# sheafint: sheaf cohomology for neural-network interpretability.

from .core import ScalableSheaf, SheafMetrics
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
    'BaseContext', 'ContextPair', 'ContextCover',
    'ParaphraseContext', 'MRPCContext', 'QQPContext', 'STSBContext', 'PAWSContext',
    'TranslationContext', 'PromptVariantContext',
    'create_standard_cover', 'create_mrpc_cover', 'create_full_cover',
    'ActivationCache', 'HookManager',
    'create_extraction_hooks', 'extract_features', 'extract_features_for_contexts',
]
