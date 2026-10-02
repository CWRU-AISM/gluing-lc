"""
CounterFact style steering: sheaf H^1 directions against CAA, random and PIXEL.

    python experiments/steering.py <experiment> --model <hf id> [flags]

Every prompt's greedy continuation is generated unsteered and under each
steering method, then scored for fact, subject and style preservation.
Results go to outputs/counterfact_steering/<model>[_pixel]_<timestamp>.json,
with every generation in the matching _full_outputs.json.
"""

from dataclasses import asdict, dataclass
from functools import partial
from typing import Literal, Optional, Tuple

import numpy as np
import torch
import tyro

from sheafint.steering import scale_to_norm
from utils.datasets import load_counterfact_data, load_mrpc_pairs
from utils.model_io import generate_with_steering, load_causal_model, n_layers, pooled_hidden_states
from utils.pixel import generate_with_pixel, mean_pooled_norm, pixel_direction, reference_vectors
from utils.results import results_path, write_json
from utils.steering import (
    contrasts, decomposition_meta, decompositions, evaluate, median_token_norm, restriction_maps,
    steering_vectors, strip_raw, summarize_method,
)
from utils.style_examples import CASUAL_EXAMPLES, FORMAL_EXAMPLES

BASELINES = ('random', 'full_style', 'mod', 'variance')


@dataclass
class Counterfact:
    """Table 4 (tab:steering_four_arch) and the fragility table: sheaf steering on CounterFact.

    Restriction maps are fit on MRPC paraphrase pairs (joint PCA, and CCA /
    contrastive maps refining it); each gets a sheaf-Laplacian and a Fisher
    decomposition. A casual-minus-formal style vector is restricted to the
    H^1 coordinates (h1_dims) or has its H^0 component removed (h0_removed).
    Baselines: random (norm-matched), full style (CAA), MoD and variance
    top-k. Every vector is scaled to --scale times the hidden-state norm.
    Methods are named {map}_{decomposition}_{h1_dims|h0_removed}; summaries
    group them per {map}_{decomposition} with the shared baselines.
    """

    model: str = 'mistralai/Mistral-7B-v0.1'
    n_tests: int = 1000
    n_pairs: int = 200
    """MRPC validation pairs that fit the restriction maps."""
    edge_dim: int = 128
    k: int = 20
    restriction_maps: Tuple[Literal['pca', 'cca', 'contrastive'], ...] = ('pca', 'cca', 'contrastive')
    methods: Tuple[str, ...] = ()
    """Subset of steering methods to run, e.g. random cca_sheaf_h1_dims (default: all)."""
    scale: float = 0.3
    norm: Literal['mean', 'median'] = 'mean'
    """Hidden-norm reference for --scale: mean-pooled state or median per-token norm."""
    batch_size: int = 16
    quantize: Literal['none', '4bit'] = '4bit'
    layer: Optional[int] = None
    """Steered block; default 0.625 * n_layers."""
    seed: int = 42
    max_new_tokens: int = 100
    output_dir: str = 'outputs/counterfact_steering'

    def run(self):
        rng = np.random.default_rng(self.seed)
        test_data = load_counterfact_data(n_samples=self.n_tests, seed=self.seed)
        mrpc_pairs = load_mrpc_pairs(n_pairs=self.n_pairs, split='validation')

        model, tokenizer = load_causal_model(self.model, self.quantize)
        layer = self.layer if self.layer is not None else int(n_layers(model) * 0.625)

        def pooled(texts):
            return pooled_hidden_states(model, tokenizer, texts, layer, batch_size=self.batch_size)

        sample_texts = FORMAL_EXAMPLES[:8] + CASUAL_EXAMPLES[:8]
        if self.norm == 'median':
            h_norm = median_token_norm(model, tokenizer, sample_texts, layer)
        else:
            h_norm = float(pooled(sample_texts).norm(dim=-1).mean())
        target_norm = h_norm * self.scale

        s1_texts = [p[0] for p in mrpc_pairs]
        s2_texts = [p[1] for p in mrpc_pairs]
        bg_texts = list(dict.fromkeys(s1_texts + s2_texts))[:self.n_pairs * 2]
        s1, s2, bg = [pooled(texts).numpy() for texts in (s1_texts, s2_texts, bg_texts)]
        style = (pooled(CASUAL_EXAMPLES).mean(0) - pooled(FORMAL_EXAMPLES).mean(0)).numpy()

        maps, map_meta = restriction_maps(s1, s2, bg, self.edge_dim, self.restriction_maps)
        decomps = decompositions(s1, s2, bg, maps, self.k)
        vectors = scale_to_norm(steering_vectors(style, decomps, rng), target_norm)
        if self.methods:
            vectors = {name: v for name, v in vectors.items() if name in self.methods}

        generators = {name: partial(generate_with_steering, model, tokenizer, steering_vector=v, layer=layer,
                                    max_new_tokens=self.max_new_tokens) for name, v in vectors.items()}
        results, texts = evaluate(model, tokenizer, test_data, generators, self.max_new_tokens)
        per_method = {name: summarize_method(r, self.seed) for name, r in results.items()}
        summaries = {}
        for group in decomps:
            members = {m: per_method[m] for m in BASELINES if m in per_method}
            members.update({m: per_method[f'{group}_{m}'] for m in ('h1_dims', 'h0_removed')
                            if f'{group}_{m}' in per_method})
            summaries[group] = members
        pairs = [(m1, m2) for m1 in ('h1_dims', 'h0_removed') for m2 in ('variance', 'random', 'full_style', 'mod')]

        out_path = results_path(self.output_dir, self.model)
        write_json(out_path, {
            'model': self.model,
            'layer': layer,
            'd': model.config.hidden_size,
            'h_norm': h_norm,
            'target_steer_norm': target_norm,
            'n_tests': len(test_data),
            'n_pairs_mrpc': len(mrpc_pairs),
            'dataset': 'counterfact',
            'restriction_maps': list(maps),
            'rmap_metadata': map_meta,
            'args': asdict(self),
            'decompositions': {group: decomposition_meta(d) for group, d in decomps.items()},
            'summaries': {group: strip_raw(s) for group, s in summaries.items()},
            'statistics': {group: contrasts(s, pairs, ('joint', 'coherent_joint')) for group, s in summaries.items()},
            'sample_outputs': texts[:50],
        })
        write_json(out_path.with_name(f'{out_path.stem}_full_outputs.json'), texts)


@dataclass
class Pixel:
    """Table 4 PIXEL rows: the PIXEL baseline (Yu et al., 2025) on CounterFact style steering.

    One dual-view casual-minus-formal direction (not per-question directions)
    is added with closed-form per-position magnitudes at the last k_tail
    prompt positions, on the prompt forward pass only: the metric is greedy
    free-form generation, so there is no continuation-position injection.
    Random (norm-matched) and full-style (end-token CAA) references are scaled
    to --scale times the mean-pooled hidden norm and added at every position.
    """

    model: str = 'meta-llama/Llama-2-7b-hf'
    n_tests: int = 1000
    quantize: Literal['none', '4bit'] = '4bit'
    layer: Optional[int] = None
    """Steered block; default 0.625 * n_layers as in counterfact."""
    seed: int = 42
    max_new_tokens: int = 100
    z_target: float = 0.85
    """Closed-form target z-score."""
    alpha_clip_lo: float = 0.0
    alpha_clip_hi: float = 200.0
    hop_alpha_max: Optional[float] = None
    """Cap on the per-position alpha."""
    scale: float = 0.3
    """Random / CAA norm as a fraction of the hidden norm."""
    output_dir: str = 'outputs/counterfact_steering'

    def run(self):
        test_data = load_counterfact_data(n_samples=self.n_tests, seed=self.seed)
        model, tokenizer = load_causal_model(self.model, self.quantize)
        model.config.use_cache = True
        layer = self.layer if self.layer is not None else int(n_layers(model) * 0.625)

        h_norm = mean_pooled_norm(model, tokenizer, FORMAL_EXAMPLES[:8] + CASUAL_EXAMPLES[:8], layer)
        direction = pixel_direction(model, tokenizer, layer)
        u = torch.from_numpy(direction.astype(np.float32))
        generators = {'pixel': partial(generate_with_pixel, model, tokenizer, layer=layer, u=u,
                                       max_new_tokens=self.max_new_tokens, z_target=self.z_target,
                                       alpha_clip=(self.alpha_clip_lo, self.alpha_clip_hi),
                                       alpha_max=self.hop_alpha_max)}
        for name, vec in reference_vectors(model, tokenizer, layer, h_norm * self.scale, self.seed).items():
            generators[name] = partial(generate_with_steering, model, tokenizer, steering_vector=vec, layer=layer,
                                       max_new_tokens=self.max_new_tokens)

        results, texts = evaluate(model, tokenizer, test_data, generators, self.max_new_tokens, desc='PIXEL')
        summary = {name: summarize_method(r, self.seed) for name, r in results.items()}
        out_path = results_path(self.output_dir, self.model, tag='pixel')
        write_json(out_path, {
            'model': self.model,
            'layer': layer,
            'd': model.config.hidden_size,
            'h_norm': h_norm,
            'n_tests': len(test_data),
            'args': asdict(self),
            'direction_norm': float(np.linalg.norm(direction)),
            'summary': strip_raw(summary),
            'statistics': contrasts(summary, [('pixel', 'random'), ('pixel', 'full_style')], ('fact',)),
            'sample_outputs': texts[:200],
        })
        write_json(out_path.with_name(f'{out_path.stem}_full_outputs.json'), texts)


EXPERIMENTS = {
    'counterfact': Counterfact,
    'pixel': Pixel,
}

if __name__ == '__main__':
    tyro.extras.subcommand_cli_from_dict(EXPERIMENTS, description=__doc__, use_underscores=True).run()
