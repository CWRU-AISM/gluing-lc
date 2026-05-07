"""
Translation contexts for cross-lingual sheaf consistency.

Pairs of parallel translations across a configurable language set; used
to test whether the H^0 subspace is preserved under language change.
"""

from typing import List
from .base import BaseContext

DEFAULT_TRANSLATIONS = [
    ("Hello", "Bonjour", "Hola", "Hallo"),
    ("Good morning", "Bonjour", "Buenos dias", "Guten Morgen"),
    ("Thank you", "Merci", "Gracias", "Danke"),
    ("I love you", "Je t'aime", "Te quiero", "Ich liebe dich"),
    ("The cat is black", "Le chat est noir", "El gato es negro", "Die Katze ist schwarz"),
    ("Water is essential for life", "L'eau est essentielle a la vie",
     "El agua es esencial para la vida", "Wasser ist lebenswichtig"),
    ("The book is on the table", "Le livre est sur la table",
     "El libro esta sobre la mesa", "Das Buch liegt auf dem Tisch"),
    ("I am learning to code", "J'apprends a coder",
     "Estoy aprendiendo a programar", "Ich lerne programmieren"),
]
DEFAULT_LANG_NAMES = ['en', 'fr', 'es', 'de']


class TranslationContext(BaseContext):
    # Pairs of parallel translations across the requested language set.

    def __init__(self, languages: List[str] = None):
        super().__init__('translation')
        self.languages = languages or DEFAULT_LANG_NAMES
        for translation in DEFAULT_TRANSLATIONS:
            for i, lang_a in enumerate(DEFAULT_LANG_NAMES):
                for j, lang_b in enumerate(DEFAULT_LANG_NAMES):
                    if i < j:
                        self.add_pair(
                            translation[i], translation[j],
                            lang_a=lang_a, lang_b=lang_b,
                        )
