# Reproducing the Paper

Commands for each table in the paper and the JSON fields behind its columns. Run them from the repository root; each run writes `outputs/<dir>/<model>_<timestamp>.json`, and `--help` on any experiment lists its flags. Gated models (Llama-2, Llama-3, Gemma-2) need `hf auth login`.

| Paper result | Command |
|---|---|
| Table 1: multi-layer grid harmonic mass | `topology.py grid` |
| Table 2: H<sup>0</sup> vs. LEACE | `retrieval.py leace` |
| Table 3: causal validation | `causal.py coordinates` |
| Table 4: CounterFact steering | `steering.py counterfact`, `steering.py pixel` |
| Retrieval baselines, restriction-map, cycle, cross-dataset and sensitivity controls | `retrieval.py baselines`, `restriction`, `cycle`, `cycle_matched`, `crossdataset`, `m_sweep`, `pooling` |
| LEACE on the 60 templated facts | `retrieval.py leace_templated` |
| Cross-lingual probe | `retrieval.py crosslingual` |
| Genuine cohomology, triangulated complexes, grid scale-up | `topology.py hypergraph`, `triangulated`, `grid --edge_dim 64` |
| Holonomy null, scalability | `topology.py holonomy_null`, `scalability` |
| Harmonic mass vs. steering fragility | `steering.py counterfact --methods random` |
| Sheaf eigenbasis ablation | `causal.py eigenbasis` |
| Dormancy and behavioural ablation | `causal.py dormancy` |

## Main Results

### Table 1: Multi-layer grid harmonic mass

```bash
for M in gpt2 microsoft/phi-2 mistralai/Mistral-7B-v0.1 Qwen/Qwen2.5-7B \
         meta-llama/Meta-Llama-3-8B meta-llama/Llama-2-7b-hf; do
  python experiments/topology.py grid --model $M
done
```

Defaults match the paper: 30 MRPC pairs, edge dim 8, layer subsets L ∈ {1, 2, 4, 8, 16} (L = 16 is skipped on GPT-2).

| Column | Field in `outputs/multilayer_hodge/<model>_k8_<ts>.json` |
|---|---|
| b<sub>1</sub> = N(L−1) | `results[i].topology.b_1` |
| Harmonic mass | `results[i].block.harmonic_fraction_total` (also `_horiz`, `_vert`) |

### Table 2: H<sup>0</sup> vs. LEACE on held-out CounterFact

```bash
python experiments/retrieval.py leace --model mistralai/Mistral-7B-v0.1
```

`outputs/leace_counterfact/<model>_<ts>.json` holds `hard_lookup_held_out` (the table, same-relation candidates) and `easy_lookup_held_out`, each with `Full`, `H0_sheaf_20d`, `LEACE_preserved`, `LEACE_preserved_20d`, `PCA_top20` and `Random_20d`.

### Table 3: Causal validation

```bash
# Protocol (b): Phi-2, Mistral-7B, Qwen-2.5-7B, OLMo-2-7B, Gemma-2-9B (fp16)
python experiments/causal.py coordinates --model microsoft/phi-2

# Protocol (a): GPT-2, Llama-2-7B, Llama-3-8B, Llama-2-13B
A="--layers half --n_dims 64 --effect logits --n_samples 1000 --pairs mixed"
python experiments/causal.py coordinates --model gpt2 $A
python experiments/causal.py coordinates --model meta-llama/Llama-2-7b-hf --quantize 4bit $A
```

| Column | Field in `outputs/causal_validation/<model>_<ts>.json` |
|---|---|
| Ratio | `experiments.layer_<l>.h1_vs_vm.ratio` |
| Layer | the `layer_<l>` key; protocol (b) reports the layer with the largest ratio |
| 95% CI | `h1_vs_vm.ci_lower`, `h1_vs_vm.ci_upper` |
| Cohen's d | `cohens_d` |

On GPT-2, protocol (b) gives ratios 12.68 / 11.10 / 2.24 at layers 3 / 6 / 9, and protocol (a) gives 18.17 at layer 6 (d = 2.34).

### Table 4: CounterFact steering

```bash
for M in meta-llama/Llama-2-7b-hf meta-llama/Meta-Llama-3-8B mistralai/Mistral-7B-v0.1 meta-llama/Llama-2-13b-hf; do
  python experiments/steering.py counterfact --model $M
done
for M in meta-llama/Llama-2-7b-hf meta-llama/Meta-Llama-3-8B mistralai/Mistral-7B-v0.1; do
  python experiments/steering.py pixel --model $M
done
```

| Row | Field in `outputs/counterfact_steering/<model>_<ts>.json` |
|---|---|
| Contrastive sheaf (H<sup>1</sup> dims) | `summaries.contrastive_sheaf.h1_dims` |
| CCA sheaf (H<sup>1</sup> dims) | `summaries.cca_sheaf.h1_dims` |
| H<sup>1</sup> dims (sheaf PCA) | `summaries.pca_sheaf.h1_dims` |
| H<sup>0</sup>-removed (sheaf) | `summaries.pca_sheaf.h0_removed` |
| Full style (CAA), Random | `summaries.pca_sheaf.full_style`, `summaries.pca_sheaf.random` |
| PIXEL | `summary.pixel` in `<model>_pixel_<ts>.json` |

Fact% is `metrics.fact.mean` (CI `ci_lo`, `ci_hi`). McNemar p-values come from the per-prompt scores in `<model>_<ts>_full_outputs.json` via `experiments/utils/statistics.py:mcnemar_test`.

## Appendix: Retrieval

Same held-out split as Table 2. The commands are for GPT-2; use `--quantize 4bit` for 7B models.

```bash
python experiments/retrieval.py baselines    --model gpt2 --quantize none
python experiments/retrieval.py restriction  --model gpt2 --quantize none
python experiments/retrieval.py cycle        --model gpt2 --quantize none
python experiments/retrieval.py crossdataset --model gpt2 --quantize none
python experiments/retrieval.py m_sweep      --model gpt2 --quantize none
python experiments/retrieval.py pooling      --model gpt2 --quantize none
```

| Table | Output | Contents |
|---|---|---|
| Retrieval baselines | `retrieval_baselines/` | easy / hard top-1 for H<sup>0</sup>, LDA, CCA, PLS, `PCA_top`, `whitened_PCA`, NCA, SimCSE, SBERT |
| Restriction-map ablation | `restriction_ablation/` | joint-PCA vs. random (5 seeds) vs. identity P |
| Cycle-aware H<sup>0</sup> | `cycle_h0/` | pair / ring / clique retrieval and b<sub>1</sub> per topology |
| Cross-dataset transfer | `crossdataset_p/` | P fit on MRPC, PAWS or QQP, scored on CounterFact |
| Dimension sensitivity | `m_sensitivity/` | m ∈ {5, 10, 20, 40, 80, 128} |
| Pooling sensitivity | `pooling_sensitivity/` | mean / last / first token |

**Matched cycle controls** (node- and edge-matched acyclic arms):

```bash
python experiments/retrieval.py cycle_matched --model gpt2                  # ring vs. acyclic, 600 facts
python experiments/retrieval.py cycle_matched --model gpt2 --variant grid   # two-layer grid, 400 facts
```

**LEACE on the 60 templated facts:**

```bash
for M in gpt2 microsoft/phi-2 Qwen/Qwen2.5-7B allenai/OLMo-2-1124-7B \
         mistralai/Mistral-7B-v0.1 google/gemma-2-9b; do
  python experiments/retrieval.py leace_templated --model $M
done
```

The table reports `easy_lookup` (top-1 against all 60 facts) in `outputs/leace_multi/<model>_<ts>.json`.

**Cross-lingual probe:**

```bash
python experiments/retrieval.py crosslingual --model meta-llama/Meta-Llama-3-8B --n_pairs 200
```

## Appendix: Topology

```bash
# Genuine cohomology on the relation hypergraph (GPT-2 at layer 8, 7B models at 16)
python experiments/topology.py hypergraph --model gpt2 --layer 8

# Triangulated complexes (edge dim 64)
python experiments/topology.py triangulated --complex grid --model gpt2
python experiments/topology.py triangulated --complex hypergraph --model gpt2

# Grid scale-up to edge dim 64 and 128
python experiments/topology.py grid --model gpt2 --edge_dim 64
python experiments/topology.py grid --model gpt2 --edge_dim 128 --n_pairs 80

# Holonomy null calibration and scalability
python experiments/topology.py holonomy_null --model gpt2
python experiments/topology.py scalability
```

| Table | Output | Columns |
|---|---|---|
| Genuine cohomology | `hypergraph_h1/` | `topology_alone.<name>.{H0,H1,H2}`; Hodge % is `sheaf_fitted.<name>.hodge_random_probe.harmonic_fraction` |
| Triangulated grid | `multilayer_triangulated/` | `results[i].topology.{b_1,b_2}`, `results[i].block.coexact_fraction_total`, `results[i].block.beta_norm2` |
| Hypergraph 1-skeleton vs. 2-complex | `hypergraph_triangulated/` | `topologies.<name>.topology_alone.{H0,H1,H2}`, `topologies.<name>.harmonic_random_unit_cochain` |
| Grid scale-up | `multilayer_hodge/` (`_k64`, `_k128`) | as Table 1 |
| Holonomy null | `holonomy_null/` | `observed`, `shuffled_null` (20 seeds), `bootstrap_N`, `identity_null` |
| Scalability | `scalability/` | `rows[i].full_ms` and `rows[i].eig_ms` per hidden size `rows[i].d` |

The hypergraph Hodge % uses an unseeded random cochain and varies in the third decimal between runs.

## Appendix: Ablation and Steering

**Harmonic mass vs. steering fragility.** Harmonic mass is the L = 2 column of Table 1; fragility comes from the random steering vector alone:

```bash
R="--methods random --n_tests 1000"
for M in mistralai/Mistral-7B-v0.1 meta-llama/Meta-Llama-3-8B meta-llama/Llama-2-7b-hf; do
  python experiments/steering.py counterfact --model $M --norm mean $R
done
python experiments/steering.py counterfact --model microsoft/phi-2 --norm mean --methods random --n_tests 200
python experiments/steering.py counterfact --model Qwen/Qwen2.5-7B --norm median $R
python experiments/steering.py counterfact --model gpt2 --norm median $R
```

Rank-correlate harmonic mass with `summaries.pca_sheaf.random.metrics.joint.mean`:

```python
from experiments.utils.statistics import spearman_exact
rho, p = spearman_exact(harmonic_mass, [-v for v in joint])   # 0.90, p = 0.014
```

**Sheaf eigenbasis ablation:**

```bash
python experiments/causal.py eigenbasis --model mistralai/Mistral-7B-v0.1   # repeat for each model in the table
```

| Column | Field in `outputs/causal_validation_sheaf/<model>_<ts>.json` |
|---|---|
| Layers | number of `experiments.layer_<l>` entries |
| H<sup>1</sup>/random, median and range | `summary.ratio_of_means` across layers |
| Prompts | minimum of `summary.frac_h1_above_max_rand` across layers |
| H<sup>0</sup>/random | median of `summary.h0_ratio_of_means` across layers |

**Dormancy and behavioural ablation:**

```bash
python experiments/causal.py dormancy --model gpt2 --n_eval 100
python experiments/causal.py dormancy --model mistralai/Mistral-7B-v0.1 --quantize 4bit --n_eval 150
```

In `outputs/dormancy_illusion/<model>_<ts>.json`, the dormancy ratio is `dormancy.H1_over_random` and recall is `summary.recall_base`, `recall_h1` and `recall_random`.

## Models and Datasets

| Model | Hugging Face ID |
|---|---|
| GPT-2 | `gpt2` |
| Phi-2 | `microsoft/phi-2` |
| Mistral-7B | `mistralai/Mistral-7B-v0.1` |
| Qwen-2.5-7B | `Qwen/Qwen2.5-7B` |
| OLMo-2-7B | `allenai/OLMo-2-1124-7B` |
| Gemma-2-9B | `google/gemma-2-9b` |
| Llama-2-7B / 13B | `meta-llama/Llama-2-7b-hf`, `meta-llama/Llama-2-13b-hf` |
| Llama-3-8B | `meta-llama/Meta-Llama-3-8B` |

| Used for | Dataset |
|---|---|
| Paraphrase pairs | `nyu-mll/glue` (`mrpc`, `qqp`), `google-research-datasets/paws` (`labeled_final`) |
| Fact retrieval and steering | `azhx/counterfact` |
| Cross-lingual probe | `Helsinki-NLP/opus-100` (`en-fr`) |

## Troubleshooting

| Symptom | Fix |
|---|---|
| `OSError: meta-llama/...` | Accept the license on the model page, then run `hf auth login` |
| Out of memory | Pass `--quantize 4bit`, or lower `--n_tests` / `--n_samples` |
| `bitsandbytes` import error | `pip install -U bitsandbytes` against your CUDA version |
| Disk full in `$HOME` | `export HF_HOME=/path/with/space` |
