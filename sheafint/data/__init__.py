"""
Data and context utilities for sheaf interpretability.

Re-exports the paraphrase / translation / prompt-variant context classes,
the cover constructors used to build a sheaf cover over them, and the
relation-template fact builder used by the holonomy null calibration.
"""

from .base import BaseContext, ContextPair
from .covers import ContextCover, create_standard_cover, create_mrpc_cover, create_full_cover
from .paraphrase import ParaphraseContext, MRPCContext, QQPContext, STSBContext, PAWSContext
from .prompts import PromptVariantContext
from .relation_templates import RELATION_TEMPLATES, ENTITIES, build_facts
from .translation import TranslationContext

__all__ = [
    'BaseContext', 'ContextPair',
    'ParaphraseContext', 'MRPCContext', 'QQPContext', 'STSBContext', 'PAWSContext',
    'TranslationContext', 'PromptVariantContext',
    'ContextCover', 'create_standard_cover', 'create_mrpc_cover', 'create_full_cover',
    'RELATION_TEMPLATES', 'ENTITIES', 'build_facts',
]
