"""
Perplexity, similarity, and sentiment helpers for comprehensive steering.

Provides a sentiment pipeline factory, a model-based perplexity estimator,
mean-pooled cosine similarity, a positive-vs-negative concept steering
direction, and small list helpers for aggregating perplexity ratios.
"""

from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F

try:
    from transformers import pipeline as _hf_pipeline
except Exception:
    _hf_pipeline = None

from .causal import device_for_inputs, hidden_states_at
from .model_io import generate as model_generate
from .model_io import generate_with_steering


def load_sentiment_pipeline():
    """
    Return a pretrained DistilBERT sentiment-analysis pipeline.

    Returns ``None`` if :mod:`transformers` cannot be imported or the model
    fails to download; callers must handle the ``None`` case.
    """
    if _hf_pipeline is None:
        return None
    try:
        return _hf_pipeline(
            'sentiment-analysis',
            model='distilbert-base-uncased-finetuned-sst-2-english',
            device=0 if torch.cuda.is_available() else -1,
        )
    except Exception:
        return None


@torch.no_grad()
def model_perplexity(model, tokenizer, text: str, max_value: float = 10000.0) -> float:
    """
    Perplexity of ``text`` under the same model used for generation.

    Returns ``float('inf')`` for trivially short inputs and clamps the
    output to ``max_value`` so single-token blowups do not dominate
    downstream aggregates.
    """
    if not text or len(text.strip()) < 3:
        return float('inf')
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=128)
    inputs = inputs.to(device_for_inputs(model))
    if inputs['input_ids'].shape[1] < 2:
        return float('inf')
    outputs = model(inputs['input_ids'], labels=inputs['input_ids'], return_dict=True)
    return float(min(np.exp(outputs.loss.item()), max_value))


def semantic_similarity(model, tokenizer, text1: str, text2: str, layer_idx: int) -> float:
    """
    Cosine similarity of mean-pooled hidden states at ``layer_idx``.

    Returns 0 for empty inputs so missing generations cannot crash the
    aggregation loop in steering experiments.
    """
    if not text1.strip() or not text2.strip():
        return 0.0
    h1 = hidden_states_at(model, tokenizer, text1, layer_idx).mean(dim=0)
    h2 = hidden_states_at(model, tokenizer, text2, layer_idx).mean(dim=0)
    return float(F.cosine_similarity(h1.unsqueeze(0), h2.unsqueeze(0)).item())


def sentiment_score(pipeline_obj, text: str) -> float:
    """
    Positive-class probability for ``text`` from a sentiment pipeline.

    Returns the neutral value 0.5 when the pipeline is unavailable, the
    text is empty, or the call raises; otherwise maps the pipeline's
    label/score pair to a [0, 1] positivity scalar.
    """
    if pipeline_obj is None or not text.strip():
        return 0.5
    try:
        result = pipeline_obj(text[:512])[0]
    except Exception:
        return 0.5
    return result['score'] if result['label'] == 'POSITIVE' else 1 - result['score']


def text_changed(baseline: str, steered: str) -> bool:
    if not baseline or not steered:
        return False
    return baseline.lower().strip() != steered.lower().strip()


def concept_steering_vector(
    model,
    tokenizer,
    positive_examples: List[str],
    negative_examples: List[str],
    layer_idx: int,
) -> torch.Tensor:
    """
    Mean-of-differences steering direction.

    Averages the layer-``layer_idx`` mean-pooled hidden states of the
    positive and negative exemplar sets, then subtracts; the resulting
    vector is added to residuals to push generations toward the positive
    concept.
    """
    pos = torch.stack([
        hidden_states_at(model, tokenizer, t, layer_idx).mean(dim=0)
        for t in positive_examples
    ]).mean(dim=0)
    neg = torch.stack([
        hidden_states_at(model, tokenizer, t, layer_idx).mean(dim=0)
        for t in negative_examples
    ]).mean(dim=0)
    return pos - neg


def perplexity_ratios(steered: List[float], baseline: List[float]) -> List[float]:
    """
    Element-wise steered/baseline perplexity ratios.

    Drops pairs with non-finite values or non-positive baselines so a
    single failed generation does not poison the median.
    """
    ratios = []
    for s, b in zip(steered, baseline):
        if np.isfinite(s) and np.isfinite(b) and b > 0:
            ratios.append(s / b)
    return ratios
