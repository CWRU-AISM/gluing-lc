# Dataset loaders shared across experiments.

from typing import List, Tuple
import numpy as np
from datasets import load_dataset


def load_mrpc_pairs(n_pairs: int = 200, split: str = 'validation') -> List[Tuple[str, str]]:
    # Paraphrase pairs (label=1) from the GLUE MRPC split.
    dataset = load_dataset('glue', 'mrpc', split=split)
    return [
        (item['sentence1'], item['sentence2'])
        for item in dataset if item['label'] == 1
    ][:n_pairs]


def load_counterfact_data(n_samples: int = 1000, seed: int = 42) -> List[dict]:
    # CounterFact entries with (subject, relation, target) and generation prompts.
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
