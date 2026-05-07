"""
Base context types for sheaf interpretability.

Defines :class:`BaseContext` and :class:`ContextPair` as the lightweight
data classes the rest of :mod:`sheafint.data` builds on.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import json


@dataclass
class ContextPair:
    # A pair of semantically equivalent texts and an optional metadata dict.
    text_a: str
    text_b: str
    context_type: str
    metadata: Optional[Dict] = None


class BaseContext:
    # Container for a list of ContextPair objects sharing a context type.

    def __init__(self, name: str):
        self.name = name
        self.pairs: List[ContextPair] = []

    def add_pair(self, text_a: str, text_b: str, **metadata):
        self.pairs.append(ContextPair(
            text_a=text_a,
            text_b=text_b,
            context_type=self.name,
            metadata=metadata,
        ))

    def get_all_texts(self) -> Tuple[List[str], List[str]]:
        # Return parallel lists of (text_a, text_b).
        return [p.text_a for p in self.pairs], [p.text_b for p in self.pairs]

    def add_from_file(self, filepath: str):
        # Append pairs from a JSON file with {text_a, text_b, metadata?} entries.
        with open(filepath, 'r') as f:
            data = json.load(f)
        for item in data:
            self.add_pair(item['text_a'], item['text_b'], **item.get('metadata', {}))

    def __len__(self):
        return len(self.pairs)
