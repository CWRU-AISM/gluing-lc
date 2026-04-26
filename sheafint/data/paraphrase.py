# Paraphrase-style context sources (hand-crafted plus GLUE/PAWS loaders).

from .base import BaseContext
try:
    from datasets import load_dataset as _load_dataset
except ImportError:
    _load_dataset = None

DEFAULT_PAIRS = [
    ("The capital of France is Paris.", "Paris is the capital city of France."),
    ("Water boils at 100 degrees Celsius.", "At 100 degrees Celsius, water reaches its boiling point."),
    ("The Earth orbits the Sun.", "Our planet Earth revolves around the Sun."),
    ("I am very happy today.", "Today I feel extremely joyful."),
    ("The movie was terrible.", "That film was absolutely awful."),
    ("She is incredibly smart.", "She has exceptional intelligence."),
    ("The cat sat on the mat.", "A feline rested upon the rug."),
    ("He walked to the store.", "He made his way to the shop on foot."),
    ("The child played in the garden.", "The kid was playing in the yard."),
    ("Democracy requires participation.", "Democratic systems need citizen involvement."),
    ("Time flies when you're having fun.", "Enjoyable moments seem to pass quickly."),
    ("Knowledge is power.", "Having knowledge gives one power."),
    ("The function returns an integer.", "An integer value is returned by this function."),
    ("Machine learning requires data.", "Data is essential for machine learning."),
    ("The algorithm has O(n) complexity.", "This algorithm runs in linear time."),
    ("The sky is blue.", "The sky is not red."),
    ("He is tall.", "He is not short."),
    ("What is your name?", "Could you tell me your name?"),
    ("How does this work?", "Can you explain how this functions?"),
    ("Why did that happen?", "What caused that to occur?"),
]


class ParaphraseContext(BaseContext):
    # Hand-crafted paraphrase pairs covering several domains.

    def __init__(self):
        super().__init__('paraphrase')
        for text_a, text_b in DEFAULT_PAIRS:
            self.add_pair(text_a, text_b)


class _HuggingfaceParaphraseContext(BaseContext):
    # Shared loader for GLUE-style paraphrase datasets.

    def __init__(self, name: str, split: str, max_pairs: int):
        super().__init__(name)
        self.split = split
        self.max_pairs = max_pairs

    def _load_fallback(self):
        for pair in ParaphraseContext().pairs[:10]:
            self.add_pair(pair.text_a, pair.text_b, source='fallback')


class MRPCContext(_HuggingfaceParaphraseContext):
    # Paraphrase pairs from Microsoft Research Paraphrase Corpus (label=1 only).

    def __init__(self, split: str = 'train', max_pairs: int = 200):
        super().__init__('mrpc', split, max_pairs)
        self._load()

    def _load(self):
        if _load_dataset is None:
            self._load_fallback()
            return
        try:
            dataset = _load_dataset('glue', 'mrpc', split=self.split)
            pairs = [
                (item['sentence1'], item['sentence2'])
                for item in dataset if item['label'] == 1
            ][:self.max_pairs]
            for text_a, text_b in pairs:
                self.add_pair(text_a, text_b, source='mrpc')
        except Exception:
            self._load_fallback()


class QQPContext(_HuggingfaceParaphraseContext):
    # Duplicate Quora question pairs (label=1).

    def __init__(self, split: str = 'train', max_pairs: int = 200):
        super().__init__('qqp', split, max_pairs)
        self._load()

    def _load(self):
        if _load_dataset is None:
            self._load_fallback()
            return
        try:
            dataset = _load_dataset('glue', 'qqp', split=self.split)
            pairs = [
                (item['question1'], item['question2'])
                for item in dataset if item['label'] == 1
            ][:self.max_pairs]
            for text_a, text_b in pairs:
                self.add_pair(text_a, text_b, source='qqp')
        except Exception:
            self._load_fallback()


class STSBContext(_HuggingfaceParaphraseContext):
    # Semantic Textual Similarity benchmark filtered by minimum score.

    def __init__(
        self,
        split: str = 'train',
        max_pairs: int = 200,
        min_similarity: float = 4.0,
    ):
        super().__init__('stsb', split, max_pairs)
        self.min_similarity = min_similarity
        self.scores = []
        self._load()

    def _load(self):
        if _load_dataset is None:
            self._load_fallback()
            return
        try:
            dataset = _load_dataset('glue', 'stsb', split=self.split)
            filtered = [
                (item['sentence1'], item['sentence2'], item['label'])
                for item in dataset if item['label'] >= self.min_similarity
            ]
            filtered.sort(key=lambda x: x[2], reverse=True)
            for text_a, text_b, score in filtered[:self.max_pairs]:
                self.add_pair(text_a, text_b, source='stsb', similarity=score)
                self.scores.append(score)
        except Exception:
            self._load_fallback()


class PAWSContext(_HuggingfaceParaphraseContext):
    # Adversarially constructed paraphrase pairs from PAWS labeled_final.

    def __init__(self, split: str = 'train', max_pairs: int = 200):
        super().__init__('paws', split, max_pairs)
        self._load()

    def _load(self):
        if _load_dataset is None:
            self._load_fallback()
            return
        try:
            dataset = _load_dataset('paws', 'labeled_final', split=self.split)
            pairs = [
                (item['sentence1'], item['sentence2'])
                for item in dataset if item['label'] == 1
            ][:self.max_pairs]
            for text_a, text_b in pairs:
                self.add_pair(text_a, text_b, source='paws')
        except Exception:
            self._load_fallback()
