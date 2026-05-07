"""
Prompt-variant contexts.

Same task, different phrasings. Used to test whether instruction-style
rewordings of the same query map to a consistent H^0 subspace.
"""

from .base import BaseContext

DEFAULT_VARIANTS = [
    [
        "What is 2 + 2?",
        "Calculate: 2 + 2",
        "2 + 2 = ?",
        "Please compute the sum of 2 and 2.",
    ],
    [
        "Summarize this text:",
        "Give me a summary of the following:",
        "TL;DR:",
        "Provide a brief summary:",
    ],
    [
        "Translate to French:",
        "Convert the following to French:",
        "Say this in French:",
        "French translation:",
    ],
    [
        "Write a Python function that",
        "Create a function in Python to",
        "Implement the following in Python:",
        "Python code for:",
    ],
    [
        "Explain how",
        "Can you explain",
        "Tell me about",
        "Describe",
    ],
]


class PromptVariantContext(BaseContext):
    # Pairs of equivalent prompt phrasings within the same task category.

    def __init__(self):
        super().__init__('prompt_variant')
        for variant_group in DEFAULT_VARIANTS:
            for i, var_a in enumerate(variant_group):
                for j, var_b in enumerate(variant_group):
                    if i < j:
                        self.add_pair(var_a, var_b)
