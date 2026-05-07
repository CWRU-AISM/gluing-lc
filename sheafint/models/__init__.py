"""
Activation extraction utilities for transformer models.

Re-exports :class:`HookManager`, :class:`ActivationCache`, the batched
``extract_features`` helpers, and per-architecture layer-name tables.
"""

from .extraction import create_extraction_hooks, extract_features, extract_features_for_contexts
from .hooks import ActivationCache, HookManager
from .layer_names import gpt2_layer_names, llama_layer_names

__all__ = [
    'ActivationCache', 'HookManager',
    'create_extraction_hooks', 'extract_features', 'extract_features_for_contexts',
    'gpt2_layer_names', 'llama_layer_names',
]
