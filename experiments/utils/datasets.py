"""
Dataset loaders shared across experiments.

All loaders pull lazily from HuggingFace and rely on ~/.cache/huggingface
unless HF_HOME / HF_DATASETS_CACHE are set.
"""

from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
from datasets import load_dataset


def load_mrpc_pairs(n_pairs: int = 200, split: str = 'validation') -> List[Tuple[str, str]]:
    """
    Paraphrase pairs from the GLUE MRPC split.

    Filters to label=1 (true paraphrases) and truncates to ``n_pairs``.
    """
    dataset = load_dataset('nyu-mll/glue', 'mrpc', split=split)
    return [
        (item['sentence1'], item['sentence2'])
        for item in dataset if item['label'] == 1
    ][:n_pairs]


def load_paraphrase_pairs(name: str, n_pairs: int = 300, seed: int = 42) -> List[Tuple[str, str]]:
    """
    Paraphrase pairs from MRPC, PAWS (labeled_final), or QQP.

    Used for cross-dataset restriction-map transfer experiments. Filters to
    label=1 (true paraphrases) and shuffles before truncating.
    """
    rng = np.random.default_rng(seed)
    if name == 'mrpc':
        ds = load_dataset('nyu-mll/glue', 'mrpc', split='train')
        pairs = [(r['sentence1'], r['sentence2']) for r in ds if r['label'] == 1]
    elif name == 'paws':
        ds = load_dataset('google-research-datasets/paws', 'labeled_final', split='train')
        pairs = [(r['sentence1'], r['sentence2']) for r in ds if r['label'] == 1]
    elif name == 'qqp':
        ds = load_dataset('nyu-mll/glue', 'qqp', split='train')
        pairs = [(r['question1'], r['question2']) for r in ds if r['label'] == 1]
    else:
        raise ValueError(f"unknown paraphrase corpus: {name}")
    rng.shuffle(pairs)
    return pairs[:n_pairs]


def load_mixed_paraphrase_pairs(max_pairs: int) -> List[Tuple[str, str]]:
    """
    Unshuffled positives from MRPC (train+validation), PAWS labeled_final
    (train+validation) and QQP (train), ``max_pairs // 3`` from each, in that
    order. Used by protocol (a) of the causal validation.
    """
    specs = [
        ('nyu-mll/glue', 'mrpc', 'train+validation', 'sentence1', 'sentence2'),
        ('google-research-datasets/paws', 'labeled_final', 'train+validation', 'sentence1', 'sentence2'),
        ('nyu-mll/glue', 'qqp', 'train', 'question1', 'question2'),
    ]
    pairs: List[Tuple[str, str]] = []
    for name, config, split, k1, k2 in specs:
        ds = load_dataset(name, config, split=split)
        pairs.extend([(r[k1], r[k2]) for r in ds if r['label'] == 1][:max_pairs // 3])
    return pairs


def load_counterfact_data(n_samples: int = 1000, seed: int = 42) -> List[dict]:
    """
    CounterFact entries with (subject, relation, target) and a single
    generation prompt. Used by the steering experiments.
    """
    ds = load_dataset('azhx/counterfact', split='train')
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(ds))[:n_samples]

    out = []
    for idx in indices:
        entry = ds[int(idx)]
        rewrite = entry['requested_rewrite']
        gen_prompts = entry['generation_prompts']
        if not gen_prompts:
            continue

        prompt_template = rewrite['prompt']
        subject = rewrite['subject']
        entity = rewrite['target_true']['str']
        fact_sentence = prompt_template.format(subject) + ' ' + entity + '.'

        out.append({
            'case_id': entry['case_id'],
            'fact': fact_sentence,
            'entity': entity,
            'subject': subject,
            'gen_prompt': gen_prompts[0],
            'all_gen_prompts': gen_prompts[:3],
            'paraphrase_prompts': entry.get('paraphrase_prompts', [])[:4],
        })
    return out


def load_counterfact_with_paraphrases(
    n_facts: int = 500,
    min_expressions: int = 4,
    seed: int = 42,
) -> List[dict]:
    """
    CounterFact entries with multiple expressions per fact, used by the
    retrieval experiments (LEACE comparison, restriction-map ablation,
    cycle-aware H^0). Each entry has ``case_id``, ``entity``, ``subject``,
    ``relation_id``, ``prompt`` (the bare CounterFact prompt, no answer) and
    ``expressions`` (list of paraphrase + generation sentences, length
    >= ``min_expressions``).
    """
    ds = load_dataset('azhx/counterfact', split='train')
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(ds))

    data = []
    for idx in indices:
        entry = ds[int(idx)]
        rw = entry['requested_rewrite']
        prompt_template = rw['prompt']
        subject = rw['subject']
        entity = rw['target_true']['str']
        relation_id = rw['relation_id']

        expressions = [prompt_template.format(subject) + ' ' + entity + '.']
        for pp in entry.get('paraphrase_prompts', [])[:6]:
            full = pp.strip()
            if not full.endswith('.'):
                full = full + ' ' + entity + '.'
            if full not in expressions:
                expressions.append(full)
        for gp in entry.get('generation_prompts', [])[:6]:
            full = gp.strip() + ' ' + entity + '.'
            if full not in expressions:
                expressions.append(full)

        if len(expressions) >= min_expressions:
            data.append({
                'case_id': entry['case_id'],
                'entity': entity,
                'subject': subject,
                'relation_id': relation_id,
                'prompt': prompt_template.format(subject),
                'expressions': expressions,
            })
        if len(data) >= n_facts:
            break
    return data


def split_facts_train_test(
    facts: List[dict],
    train_frac: float = 0.7,
    seed: int = 42,
) -> Tuple[List[int], List[int]]:
    """
    Random train/test split over fact ids.

    The seeded permutation makes the split reproducible across the LEACE,
    restriction-map, and pooling experiments so they all see identical
    held-out facts.
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(facts))
    n_train = int(train_frac * len(facts))
    return list(map(int, perm[:n_train])), list(map(int, perm[n_train:]))


def build_centroid_query_split(
    facts: List[dict],
    test_ids: List[int],
    seed: int = 42,
) -> Dict[str, list]:
    """
    For each test fact, randomly partition its expressions in half for
    centroids vs queries. Returns dict with parallel lists:
    centroid_prompts, centroid_fact_id, query_prompts, query_fact_id,
    query_relation_id.
    """
    rng = np.random.default_rng(seed)
    centroid_prompts: List[str] = []
    centroid_fact_id: List[int] = []
    query_prompts: List[str] = []
    query_fact_id: List[int] = []
    query_relation_id: List[str] = []

    for i in test_ids:
        f = facts[i]
        exprs = f['expressions']
        order = rng.permutation(len(exprs))
        n_cent = max(1, len(exprs) // 2)
        for k in order[:n_cent]:
            centroid_prompts.append(exprs[k])
            centroid_fact_id.append(i)
        for k in order[n_cent:]:
            query_prompts.append(exprs[k])
            query_fact_id.append(i)
            query_relation_id.append(f['relation_id'])

    return {
        'centroid_prompts': centroid_prompts,
        'centroid_fact_id': np.array(centroid_fact_id),
        'query_prompts': query_prompts,
        'query_fact_id': np.array(query_fact_id),
        'query_relation_id': query_relation_id,
    }


def hard_restrict_by_relation(
    facts: List[dict],
    test_ids: List[int],
    query_relation_id: List[str],
) -> List[List[int]]:
    """
    Per-query candidate set: all test facts that share its relation. Used
    for hard same-relation retrieval restriction.
    """
    by_relation: Dict[str, List[int]] = defaultdict(list)
    for i in test_ids:
        by_relation[facts[i]['relation_id']].append(i)
    return [by_relation[r] for r in query_relation_id]


def load_translation_pairs(
    n_pairs: int = 200,
    seed: int = 42,
    min_words: int = 4,
    max_words: int = 40,
) -> Tuple[List[str], List[str]]:
    """
    EN<->FR sentence-aligned pairs from the OPUS-100 test split, used by the
    cross-lingual invariance probe. Returns ``(en, fr)`` parallel lists.
    """
    ds = load_dataset('Helsinki-NLP/opus-100', 'en-fr', split='test')
    rng = np.random.default_rng(seed)
    en: List[str] = []
    fr: List[str] = []
    for i in rng.permutation(len(ds)):
        t = ds[int(i)]['translation']
        e, f = t['en'].strip(), t['fr'].strip()
        if min_words <= len(e.split()) <= max_words and min_words <= len(f.split()) <= max_words:
            en.append(e)
            fr.append(f)
        if len(en) >= n_pairs:
            break
    return en, fr
