#!/usr/bin/env python3
"""
Comprehensive steering metrics (Table 3).

Reports perplexity ratios, mean-pooled cosine similarity, sentiment
shift, and text-change rates for H^0 / H^1 / PCA-projected steering
directions on a casual-vs-formal style axis.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List
import numpy as np
import torch
from scipy import stats as scipy_stats
from sklearn.decomposition import PCA
from tqdm import tqdm
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.utils.causal import hidden_states_at
from experiments.utils.datasets import load_mrpc_pairs
from experiments.utils.model_io import (
    generate as model_generate,
    generate_with_steering,
    load_causal_model,
    n_layers as model_n_layers,
)
from experiments.utils.perplexity import (
    concept_steering_vector,
    load_sentiment_pipeline,
    model_perplexity,
    perplexity_ratios,
    semantic_similarity,
    sentiment_score,
    text_changed,
)
from experiments.utils.statistics import bootstrap_ci

POSITIVE_EXAMPLES = [
    "This is absolutely wonderful and amazing!",
    "I love this so much, it's fantastic!",
    "What a great day, everything is perfect!",
    "This makes me so happy and excited!",
    "Excellent work, I'm very impressed!",
    "Outstanding performance, truly remarkable!",
    "I'm delighted with the results!",
    "This exceeded all my expectations!",
]
NEGATIVE_EXAMPLES = [
    "This is terrible and disappointing.",
    "I hate this, it's awful.",
    "What a bad day, everything went wrong.",
    "This makes me sad and frustrated.",
    "Poor work, I'm very disappointed.",
    "Terrible performance, utterly dismal.",
    "I'm upset with the results.",
    "This fell short of expectations.",
]
PROMPT_TEMPLATES = [
    "The movie was", "I think this restaurant is", "My experience with the product was",
    "The service at the hotel was", "This book is", "The weather today makes me feel",
    "My opinion on this matter is", "The presentation was", "I found the lecture to be",
    "The new policy seems", "Working here has been", "The concert was", "This meal is",
    "The app update is", "My trip was", "The customer service was", "This game is",
    "The meeting went", "I think the solution is", "The news was", "The atmosphere is",
    "My impression of the city is", "The quality seems", "This experience was",
    "The team performance was", "I believe the outcome is", "The design looks",
    "My feeling about this is", "The results appear", "This approach seems",
]


def _pca_steering_vectors(diff_vectors: np.ndarray, style_np: np.ndarray, n_dims: int):
    # Build PCA-high and PCA-low projection vectors for the style direction.
    pca = PCA(n_components=min(64, len(diff_vectors)))
    pca.fit(diff_vectors)

    high = np.zeros_like(style_np)
    for i in range(min(n_dims, pca.n_components_)):
        coef = float(np.dot(style_np, pca.components_[i]))
        high += coef * pca.components_[i]

    low = np.zeros_like(style_np)
    for i in range(max(0, len(pca.components_) - n_dims), len(pca.components_)):
        coef = float(np.dot(style_np, pca.components_[i]))
        low += coef * pca.components_[i]
    return high, low, pca.explained_variance_ratio_


def _run_steering_pass(model, tokenizer, prompts, vectors, layer):
    # Generate baseline and steered outputs for every prompt and steering vector.
    metrics = {
        'baseline': {'ppl': [], 'sentiment': []},
    }
    for name in vectors:
        metrics[name] = {'ppl': [], 'sentiment': [], 'similarity': [], 'changed': []}

    sentiment_pipe = load_sentiment_pipeline()

    for prompt in tqdm(prompts, desc='Evaluating'):
        try:
            baseline = model_generate(model, tokenizer, prompt, max_new_tokens=30)
            metrics['baseline']['ppl'].append(model_perplexity(model, tokenizer, baseline))
            metrics['baseline']['sentiment'].append(sentiment_score(sentiment_pipe, baseline))

            for name, vec in vectors.items():
                steered = generate_with_steering(model, tokenizer, prompt, vec, layer, max_new_tokens=30)
                metrics[name]['ppl'].append(model_perplexity(model, tokenizer, steered))
                metrics[name]['sentiment'].append(sentiment_score(sentiment_pipe, steered))
                metrics[name]['similarity'].append(
                    semantic_similarity(model, tokenizer, baseline, steered, model_n_layers(model) - 1)
                )
                metrics[name]['changed'].append(int(text_changed(baseline, steered)))
        except Exception:
            continue
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Comprehensive steering evaluation')
    parser.add_argument('--model', type=str, default='gpt2')
    parser.add_argument('--quantize', type=str, default='none')
    parser.add_argument('--n_samples', type=int, default=500)
    parser.add_argument('--scale', type=float, default=2.0)
    args = parser.parse_args()

    model, tokenizer = load_causal_model(args.model, args.quantize)
    layer = model_n_layers(model) * 2 // 3

    pairs = load_mrpc_pairs(n_pairs=100, split='validation')
    diffs: List[torch.Tensor] = []
    diff_vectors: List[np.ndarray] = []
    for s1, s2 in tqdm(pairs, desc='Computing H0/H1'):
        try:
            h1 = hidden_states_at(model, tokenizer, s1, layer).mean(dim=0)
            h2 = hidden_states_at(model, tokenizer, s2, layer).mean(dim=0)
            diffs.append((h1 - h2).abs())
            diff_vectors.append((h1 - h2).cpu().float().numpy())
        except Exception:
            continue
    if not diffs:
        return

    mean_diff = torch.stack(diffs).mean(dim=0).cpu().float().numpy()
    sorted_dims = np.argsort(mean_diff)
    h0_dims = sorted_dims[:20].tolist()
    h1_dims = sorted_dims[-20:][::-1].tolist()

    sentiment_vector = concept_steering_vector(
        model, tokenizer, POSITIVE_EXAMPLES, NEGATIVE_EXAMPLES, layer,
    )
    style_np = sentiment_vector.cpu().float().numpy()
    pca_high, pca_low, explained = _pca_steering_vectors(np.stack(diff_vectors), style_np, n_dims=20)

    h0_only = torch.zeros_like(sentiment_vector)
    h0_only[h0_dims] = sentiment_vector[h0_dims]
    h1_only = torch.zeros_like(sentiment_vector)
    h1_only[h1_dims] = sentiment_vector[h1_dims]
    pca_high_t = torch.tensor(pca_high, dtype=sentiment_vector.dtype, device=sentiment_vector.device)
    pca_low_t = torch.tensor(pca_low, dtype=sentiment_vector.dtype, device=sentiment_vector.device)

    vectors = {
        'h0': h0_only * args.scale,
        'h1': h1_only * args.scale,
        'pca_high': pca_high_t * args.scale,
        'pca_low': pca_low_t * args.scale,
    }

    prompts = [PROMPT_TEMPLATES[i % len(PROMPT_TEMPLATES)] for i in range(args.n_samples)]
    metrics = _run_steering_pass(model, tokenizer, prompts, vectors, layer)

    summary = {
        'perplexity_ratio': {
            name: bootstrap_ci(perplexity_ratios(metrics[name]['ppl'], metrics['baseline']['ppl']))
            for name in vectors
        },
        'text_change': {name: bootstrap_ci(metrics[name]['changed']) for name in vectors},
        'similarity': {name: bootstrap_ci(metrics[name]['similarity']) for name in vectors},
    }

    h1_vs_h0_change = scipy_stats.mannwhitneyu(
        metrics['h1']['changed'], metrics['h0']['changed'], alternative='greater',
    )
    h1_vs_pca_change = scipy_stats.mannwhitneyu(
        metrics['h1']['changed'], metrics['pca_high']['changed'], alternative='greater',
    )
    h0_vs_pca_sim = scipy_stats.mannwhitneyu(
        metrics['h0']['similarity'], metrics['pca_low']['similarity'], alternative='greater',
    )

    output_dir = Path('outputs/comprehensive_steering')
    output_dir.mkdir(parents=True, exist_ok=True)
    slug = args.model.replace('/', '_')

    with open(output_dir / f'{slug}_n{args.n_samples}.json', 'w') as f:
        json.dump({
            'model': args.model,
            'layer': layer,
            'n_samples': args.n_samples,
            'steering_scale': args.scale,
            'h0_dims': h0_dims,
            'h1_dims': h1_dims,
            'pca_explained_variance': explained.tolist(),
            'metrics': summary,
            'hypothesis_tests': {
                'h1_vs_h0_change': {'p_value': float(h1_vs_h0_change.pvalue),
                                    'significant': bool(h1_vs_h0_change.pvalue < 0.05)},
                'h1_vs_pca_change': {'p_value': float(h1_vs_pca_change.pvalue),
                                     'significant': bool(h1_vs_pca_change.pvalue < 0.05)},
                'h0_vs_pca_sim': {'p_value': float(h0_vs_pca_sim.pvalue),
                                  'significant': bool(h0_vs_pca_sim.pvalue < 0.05)},
            },
            'timestamp': datetime.now().isoformat(),
        }, f, indent=2)


if __name__ == '__main__':
    main()
