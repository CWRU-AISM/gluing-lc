# Gluing Local Contexts into Global Meaning: A Sheaf-Theoretic Decomposition of Transformer Representations

[![Project Page](https://img.shields.io/badge/Project-Website-blue)](https://cwru-aism.github.io/gluing-lc-page/)
[![Paper](https://img.shields.io/badge/Paper-PDF-red)](https://cwru-aism.github.io/gluing-lc-page/static/paper.pdf)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Authors:** Bryce Grant, Peng Wang

## Overview

This repository contains the reference implementation for the paper above. We treat a transformer's per-context representations as the stalks of a sheaf over a cover of semantically equivalent inputs and study its cohomology:

* `H^0` captures features that are consistent across paraphrases (the "monosemantic" subspace).
* `H^1` captures features that vary with phrasing (the "polysemantic" subspace, useful for steering).
* The sheaf Laplacian gives a spectral handle on both, with chain-complex exactness guaranteed by construction.

The package ships a scalable sheaf implementation, restriction-map learners (joint PCA, Fisher / contrastive), and end-to-end steering experiments on CounterFact and MRPC across GPT-2, Llama-2, Llama-3, and Mistral.

## Repository layout

```
sheaf_final/
  sheafint/
    core/        ScalableSheaf, projections, coboundaries, cohomology, spectrum
    data/        Paraphrase, translation, prompt-variant contexts and covers
    models/      HookManager, batched activation extraction, layer-name tables
    steering/    Restriction maps, sheaf/Fisher decompositions, steering vectors
    baselines.py Cosine, CKA, MSE consistency baselines
  experiments/
    run_causal_validation.py
    run_cohomology_validation.py
    run_counterfact_steering.py
    run_comprehensive_steering.py
    utils/       Shared dataset, model, statistics, and metric helpers
  figures/       Shared figure style for camera-ready plots
  tests/         Pytest smoke tests
```

## Setup

The codebase targets Python 3.10+ on Linux with CUDA-capable GPUs for the LLM experiments.

```bash
conda env create -f environment.yml
conda activate sheaf_final
pip install -e ".[llm,test]"
```

## Models and data

### Models

Models are loaded via `transformers.AutoModelForCausalLM.from_pretrained(name, ...)` and are downloaded from the HuggingFace Hub on first use. The four checkpoints we report on are:

| Name in CLI | HuggingFace ID | License / access |
|---|---|---|
| GPT-2 | `gpt2` | Open, no auth needed |
| Llama-2-7B | `meta-llama/Llama-2-7b-hf` | Gated, requires HF token + Meta license acceptance |
| Llama-3-8B | `meta-llama/Meta-Llama-3-8B` | Gated, requires HF token + Meta license acceptance |
| Mistral-7B | `mistralai/Mistral-7B-v0.1` | Open, but rate-limited without a token |

To use the gated models:

```bash
hf auth login                   # paste a token from https://huggingface.co/settings/tokens
# or set the environment variable directly
export HF_TOKEN=hf_xxx
```

You also need to accept each model's license on its HuggingFace page once per account.

The 7B+ checkpoints are too large for a single 24 GB consumer GPU at fp16, so the steering scripts default to 4-bit quantization via `bitsandbytes`:

```bash
python experiments/run_counterfact_steering.py \
    --model mistralai/Mistral-7B-v0.1 --quantize 4bit
```

Pass `--quantize none` to load in fp16 if you have at least an A100/H100. Downloads are cached under `~/.cache/huggingface/hub` by default; override with `HF_HOME` or `TRANSFORMERS_CACHE` if you have limited disk in `$HOME`.

### Data

All datasets are pulled lazily by `datasets.load_dataset` on first run and cached under `~/.cache/huggingface/datasets` (override with `HF_DATASETS_CACHE`). No manual download is required.

| Where it's used | HuggingFace dataset |
|---|---|
| MRPC paraphrase pairs (sheaf construction) | `glue` config `mrpc` |
| QQP duplicate questions (optional cover) | `glue` config `qqp` |
| STS-B similarity (optional) | `glue` config `stsb` |
| PAWS adversarial paraphrases (optional) | `paws` config `labeled_final` |
| CounterFact factual steering targets | `azhx/counterfact` |

GLUE downloads are public; `azhx/counterfact` is also public on the Hub. If you are on a network without outbound access, pre-fetch them on a connected machine and copy the cache directory over, or pass `HF_DATASETS_OFFLINE=1` after the cache is warm.

## Running experiments

All experiments write JSON outputs under `outputs/<experiment-name>/` from the working directory. Run them from the `sheaf_final/` root.

### Causal validation (Table 1)

```bash
python experiments/run_causal_validation.py --model gpt2 --n_samples 100
python experiments/run_causal_validation.py \
    --model mistralai/Mistral-7B-v0.1 --quantize 4bit --n_samples 200
```

Compares H1 ablations to variance-matched dimension controls and reports bootstrap CIs on the resulting effect ratios.

### Cohomology vs. variance heuristic

```bash
python experiments/run_cohomology_validation.py --model gpt2 --n_pairs 100
```

Fits a `ScalableSheaf` on MRPC paraphrase activations, runs the Laplacian spectrum, and correlates the H0 / H1 dim choices with the per-dimension variance heuristic.

### CounterFact steering (Table 4)

```bash
python experiments/run_counterfact_steering.py \
    --model mistralai/Mistral-7B-v0.1 --n_tests 1000 --quantize 4bit
```

Builds joint PCA restriction maps, sheaf-Laplacian and Fisher decompositions, and evaluates steering with random / full-style / variance / H1-dim / H0-removed vectors using the McNemar paired test.

### Comprehensive steering metrics (Table 3)

```bash
python experiments/run_comprehensive_steering.py --model gpt2 --n_samples 500
```

Reports perplexity ratios, semantic similarity, and text-change rates for H0 vs. H1 vs. PCA-projected steering directions.

## Library quick start

```python
import torch
from sheafint import ScalableSheaf, ParaphraseContext

ctx = ParaphraseContext()
texts_a, texts_b = ctx.get_all_texts()

# Replace these with extracted hidden states for `texts_a` / `texts_b`.
features_a = torch.randn(len(texts_a), 768)
features_b = features_a + 0.05 * torch.randn_like(features_a)

sheaf = ScalableSheaf(edge_dim=16, device='cpu')
sheaf.fit(
    {'a': features_a, 'b': features_b},
    {('a', 'b'): torch.arange(len(texts_a))},
)
metrics = sheaf.evaluate(
    {'a': features_a, 'b': features_b},
    {('a', 'b'): torch.arange(len(texts_a))},
)
spectrum = sheaf.compute_laplacian_spectrum()
print(metrics.consistency_energy, spectrum['h0_dim'])
```

For full HuggingFace pipelines see `sheafint.models.extract_features` and `experiments/utils/model_io.py`.

## Reproducing the paper

The four scripts above produce the headline tables and steering numbers when run with the default args on each of GPT-2, Llama-2-7B, Llama-3-8B, and Mistral-7B. Aggregating across models is done outside this package using whatever bookkeeping you prefer; the JSON outputs already contain bootstrap CIs and significance tests.

## Citation

```
@inproceedings{grant2026gluing,
  title     = {Gluing Local Contexts into Global Meaning: A Sheaf-Theoretic Decomposition of Transformer Representations},
  author    = {Bryce Grant and Peng Wang},
  booktitle = {ICLR 2026 Workshop on Unifying Concept Representation Learning},
  year      = {2026},
  url       = {https://openreview.net/forum?id=eub5YrhExo}
}
```

## License

Released under the [MIT License](LICENSE).
