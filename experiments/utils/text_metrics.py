# Lightweight text quality metrics for generation evaluation.

from collections import Counter


def first_sentence(text: str) -> str:
    # Trim to the first sentence using simple punctuation cues.
    text = text.strip()
    if not text:
        return ''
    for sep in ('. ', '.\n', '!\n', '! ', '?\n', '? '):
        if sep in text:
            return text[:text.index(sep) + 1].strip()
    return text[:200].strip()


def is_degenerate(text: str, max_ngram: int = 4) -> bool:
    # Detect repetitive n-gram loops within the first sentence.
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
    # Treat outputs as "changed" when their Jaccard word overlap is below threshold.
    if baseline.strip() == steered.strip():
        return False
    a = set(baseline.lower().split())
    b = set(steered.lower().split())
    union = len(a | b)
    if union == 0:
        return False
    return (len(a & b) / union) < jaccard_threshold


def evaluate_generation(baseline: str, steered: str, entity: str, subject: str) -> dict:
    # Combine fact / subject / style / coherence checks into a single result dict.
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
