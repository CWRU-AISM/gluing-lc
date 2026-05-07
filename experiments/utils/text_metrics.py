"""
Lightweight text-quality metrics for generation evaluation.

Used by the steering experiments to detect repetitive degeneration, check
whether the original entity / subject survives steering, and judge whether
a generation has changed meaningfully relative to the baseline.
"""

from collections import Counter


def first_sentence(text: str) -> str:
    """
    Trim ``text`` to its first sentence using simple punctuation cues.

    Falls back to the first 200 characters when no terminator is found so
    extremely long single-sentence outputs do not propagate downstream.
    """
    text = text.strip()
    if not text:
        return ''
    for sep in ('. ', '.\n', '!\n', '! ', '?\n', '? '):
        if sep in text:
            return text[:text.index(sep) + 1].strip()
    return text[:200].strip()


def is_degenerate(text: str, max_ngram: int = 4) -> bool:
    """
    Detect repetitive n-gram loops within the first sentence.

    An output is flagged as degenerate when any n-gram of length 2..N
    occurs at least three times and accounts for more than 30% of the
    n-grams in the first sentence.
    """
    words = text.lower().split()
    if len(words) < 10:
        return False
    first_words = first_sentence(text).lower().split()
    if len(first_words) < 8:
        return False
    for n in range(2, max_ngram + 1):
        ngrams = [tuple(first_words[i:i + n]) for i in range(len(first_words) - n + 1)]
        if not ngrams:
            continue
        most_common = Counter(ngrams).most_common(1)[0][1]
        if most_common >= 3 and most_common / len(ngrams) > 0.3:
            return True
    return False


def entity_in_text(text: str, entity: str) -> bool:
    return entity.lower() in text.lower()


def subject_in_text(text: str, subject: str) -> bool:
    if subject.lower() in text.lower():
        return True
    parts = subject.split()
    if len(parts) > 1:
        return parts[-1].lower() in text.lower()
    return False


def text_meaningfully_changed(baseline: str, steered: str, jaccard_threshold: float = 0.8) -> bool:
    """
    Whether ``steered`` has departed from ``baseline``.

    Treats the outputs as different when their Jaccard word overlap drops
    below ``jaccard_threshold``; identical-after-stripping outputs are
    rejected up front.
    """
    if baseline.strip() == steered.strip():
        return False
    a = set(baseline.lower().split())
    b = set(steered.lower().split())
    union = len(a | b)
    if union == 0:
        return False
    return (len(a & b) / union) < jaccard_threshold


def evaluate_generation(baseline: str, steered: str, entity: str, subject: str) -> dict:
    """
    Combine fact / subject / style / coherence checks into a single dict.

    The ``joint`` field requires both the entity to survive and the style
    to change; ``coherent_joint`` additionally requires the steered output
    not to be degenerate.
    """
    fact_preserved = entity_in_text(steered, entity)
    subject_preserved = subject_in_text(steered, subject)
    style_changed = text_meaningfully_changed(baseline, steered)
    coherent = not is_degenerate(steered)

    joint = fact_preserved and style_changed
    coherent_joint = joint and coherent

    return {
        'fact': int(fact_preserved),
        'subject': int(subject_preserved),
        'style': int(style_changed),
        'coherent': int(coherent),
        'joint': int(joint),
        'coherent_joint': int(coherent_joint),
    }
