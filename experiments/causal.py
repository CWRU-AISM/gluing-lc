"""
Causal role of H^0 / H^1 directions: ablate them and measure the change in model output.

    python experiments/causal.py <experiment> [flags]

Results go to outputs/<dir>/<model>_<timestamp>.json.
"""

import time
from dataclasses import dataclass
from functools import partial
from typing import Literal, Optional, Tuple

import numpy as np
import torch
import tyro
from tqdm import tqdm

from sheafint import fit_h0_pca
from sheafint.core.h0 import joint_pca_projection
from utils.causal import (
    ProjectionAblator, ablation_effect, bootstrap_effect_ratio, greedy_generation, haar_basis,
    identify_h0_h1_dims, kl_divergence, last_token_logits, projection_spread, variance_matched_dims,
)
from utils.datasets import load_mixed_paraphrase_pairs, load_mrpc_pairs
from utils.eigenbasis import default_layers, energy, pooled_block_outputs, summarize, versus
from utils.model_io import get_layers, load_causal_model, n_layers, pooled_hidden_states
from utils.results import results_path, write_json
from utils.retrieval import HeldOutConfig, fit_sheaf_h1_basis, load_heldout_facts, make_train_pairs


@dataclass
class Coordinates:
    """Causal validation table: H^1 coordinate ablation vs variance-matched controls.

    Per layer, ranks residual coordinates by mean |within-pair difference| over
    paraphrase pairs, takes the bottom / top --n_dims as H^0 / H^1, zeros each
    set in the layer output and measures the change in model output. The
    control is the --n_dims coordinates (excluding H^0 and H^1) whose per-dim
    variability is nearest the mean per-dim variability of H^1. Cohen's d uses
    the pooled np.std (ddof 0).

    Protocol (b), the defaults: MRPC validation positives (first 100 pairs),
    dims fit on the first 50, test texts are both sentences of each pair,
    unpaired bootstrap. Protocol (a) (layer L/2, 64 dims, final-position
    logits, 1000 sentences, mixed pairs): 2 * n_samples pairs, with
    n_samples * 2 // 3 positives each from MRPC train+validation, PAWS
    labeled_final and QQP train; dims fit on the first 200 pairs in the model
    dtype; test texts are the first sentence of the first n_samples pairs;
    paired bootstrap.
    """

    model: str = 'gpt2'
    quantize: Literal['none', '4bit'] = 'none'
    n_samples: int = 100
    n_dims: int = 20
    layers: Literal['quartiles', 'half'] = 'quartiles'
    """quartiles = L/4, L/2, 3L/4; half = L/2."""
    effect: Literal['hidden', 'logits'] = 'hidden'
    """hidden = L2 of the final-layer mean-pooled state; logits = L2 of the final-position logits."""
    pairs: Literal['mrpc', 'mixed'] = 'mrpc'
    """Data and bootstrap of protocol (b) or (a)."""
    n_bootstrap: int = 1000
    output_dir: str = 'outputs/causal_validation'

    def run(self):
        model, tokenizer = load_causal_model(self.model, self.quantize)
        if self.pairs == 'mixed':
            pairs = load_mixed_paraphrase_pairs(self.n_samples * 2)
            texts = [p[0] for p in pairs[:self.n_samples]]
        else:
            pairs = load_mrpc_pairs(n_pairs=100)
            texts = [s for pair in pairs for s in pair][:self.n_samples]

        n_total = n_layers(model)
        layers = [n_total // 4, n_total // 2, 3 * n_total // 4] if self.layers == 'quartiles' else [n_total // 2]
        experiments = {}
        for layer in layers:
            result = self._layer(model, tokenizer, layer, pairs, texts)
            if result is not None:
                experiments[f'layer_{layer}'] = result
        write_json(results_path(self.output_dir, self.model), {
            'model': self.model,
            'n_samples': len(texts),
            'protocol': {k: getattr(self, k) for k in ('layers', 'n_dims', 'effect', 'pairs', 'quantize')},
            'experiments': experiments,
        })

    def _layer(self, model, tokenizer, layer, pairs, texts):
        """H^0 / H^1 / random / variance-matched ablation effects at one layer."""
        mixed = self.pairs == 'mixed'
        h0_dims, h1_dims, variance = identify_h0_h1_dims(
            model, tokenizer, pairs[:200 if mixed else 50], layer, n_dims=self.n_dims, model_dtype_diffs=mixed)
        if not h0_dims or not h1_dims:
            return None

        h1_variance = float(variance[h1_dims].sum())
        random_dims = np.random.choice(model.config.hidden_size, size=self.n_dims, replace=False).tolist()
        # protocol (a) targets n_dims * the fp16 mean of the H1 variabilities
        target = float(np.mean(variance[h1_dims])) * self.n_dims if mixed else h1_variance
        vm_dims = variance_matched_dims(target, variance, self.n_dims, exclude_dims=h0_dims + h1_dims)

        n_total = n_layers(model)
        conditions = {'h0': h0_dims, 'h1': h1_dims, 'random': random_dims, 'variance_matched': vm_dims}
        l2 = {name: [] for name in conditions}
        for text in tqdm(texts, desc=f'Ablation L{layer}'):
            for name, dims in conditions.items():
                try:
                    effect = ablation_effect(model, tokenizer, text, layer, n_total, dims, self.effect)
                except Exception:
                    continue
                if not np.isnan(effect['l2_distance']):
                    l2[name].append(effect['l2_distance'])

        def stats(name):
            return {'mean': np.mean(l2[name]), 'std': np.std(l2[name])}

        h1, vm = np.asarray(l2['h1']), np.asarray(l2['variance_matched'])
        return {
            'layer': layer,
            'h0_dims': h0_dims,
            'h1_dims': h1_dims,
            'h0_variance': float(variance[h0_dims].sum()),
            'h1_variance': h1_variance,
            'vm_variance': float(variance[vm_dims].sum()),
            'h0_effect': stats('h0'),
            'h1_effect': stats('h1'),
            'random_effect': stats('random'),
            'vm_effect': stats('variance_matched'),
            'h1_vs_vm': bootstrap_effect_ratio(h1, vm, self.n_bootstrap, paired=mixed),
            'h1_vs_h0': bootstrap_effect_ratio(l2['h1'], l2['h0'], self.n_bootstrap, paired=mixed),
            'cohens_d': (h1.mean() - vm.mean()) / np.sqrt((h1.std() ** 2 + vm.std() ** 2) / 2),
        }


@dataclass
class Eigenbasis:
    """Sheaf eigenbasis ablation (tab:sheaf_basis): project out H^1 vs random subspaces.

    Per block l: H^1 (top-k eigenvectors of L_F, joint-PCA restriction map) and
    H^0 (bottom-k) are fit on mean-pooled block-l outputs of MRPC train
    paraphrase pairs. On held-out MRPC sentences the block-l output is
    projected onto span(S)^perp at every position, the rest of the forward pass
    runs, and the effect is KL(p(Y) || p(Y | do)) of the next-token
    distribution at the final prompt position. Controls, --n_controls draws
    each where random: Haar-uniform k-dim subspaces of R^d, random k-dim
    subspaces of span(P), and the top-k principal directions of the fit
    activations. The last two remove as much activation energy as H^1 or more.
    """

    model: str = 'gpt2'
    quantize: Literal['none', '4bit'] = 'none'
    layers: Optional[Tuple[int, ...]] = None
    """0-indexed blocks; default 1, 3, 5, ..., n_layers - 1."""
    k: int = 20
    """Subspace dimension."""
    edge_dim: int = 128
    n_fit_pairs: int = 200
    n_eval: int = 100
    n_controls: int = 20
    n_boot: int = 1000
    batch_size: int = 16
    seed: int = 42
    output_dir: str = 'outputs/causal_validation_sheaf'

    def run(self):
        out_path = results_path(self.output_dir, self.model)
        pairs = load_mrpc_pairs(n_pairs=self.n_fit_pairs + self.n_eval, split='train')
        fit_pairs = pairs[:self.n_fit_pairs]
        eval_prompts = [a for a, _ in pairs[self.n_fit_pairs:]]
        if len(eval_prompts) != self.n_eval:
            raise ValueError(f'only {len(eval_prompts)} MRPC eval sentences')

        model, tokenizer = load_causal_model(self.model, quantize=self.quantize)
        tokenizer.padding_side = 'right'  # last_token_logits reads the final non-pad position
        device = next(model.parameters()).device
        layers = list(self.layers) if self.layers else default_layers(n_layers(model))
        rng = np.random.default_rng(self.seed)
        bs = self.batch_size

        acts = pooled_block_outputs(model, tokenizer, [a for a, _ in fit_pairs] + [b for _, b in fit_pairs], layers, bs)
        batches = [eval_prompts[i:i + bs] for i in range(0, len(eval_prompts), bs)]
        base = [last_token_logits(model, tokenizer, b, device).cpu() for b in batches]
        results = {
            f'layer_{layer}': self._layer(model, tokenizer, layer, acts[layer][:self.n_fit_pairs],
                                          acts[layer][self.n_fit_pairs:], batches, base, rng)
            for layer in layers
        }
        write_json(out_path, {
            'model': self.model, 'quantize': self.quantize, 'k': self.k, 'edge_dim': self.edge_dim,
            'n_fit_pairs': self.n_fit_pairs, 'n_eval': self.n_eval, 'n_controls': self.n_controls,
            'seed': self.seed, 'layers': layers, 'experiments': results,
        })

    def _layer(self, model, tokenizer, layer, H_a, H_b, batches, base, rng):
        """KL under H^1, H^0 and the three control families at one block."""
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        d, k = H_a.shape[1], self.k
        t0 = time.time()
        U_h1, h1_eigvals = fit_sheaf_h1_basis(H_a, H_b, k, self.edge_dim)
        U_h0 = fit_h0_pca(H_a, H_b, k=k, edge_dim=self.edge_dim)

        X = torch.cat([H_a, H_b]).numpy().astype(np.float64)
        S2 = X.T @ X / len(X)  # uncentred: the projection removes raw h U U^T
        U_pca = np.linalg.svd(X - X.mean(0), full_matrices=False)[2][:k].T
        P = joint_pca_projection(H_a, H_b, self.edge_dim).astype(np.float64)

        ablator = ProjectionAblator(U_h1, device, dtype)
        ablator.enabled = True
        handle = get_layers(model)[layer].register_forward_hook(ablator)

        def kl_under(U, prompts, base_logits):
            ablator.U = torch.from_numpy(np.ascontiguousarray(U)).to(device=device, dtype=dtype)
            return kl_divergence(base_logits, last_token_logits(model, tokenizer, prompts, device).cpu())

        kl_h1, kl_h0, kl_rand, kl_pca, kl_spanp, e_rand, e_spanp = [], [], [], [], [], [], []
        try:
            for prompts, base_logits in zip(batches, base):
                kl_h1 += kl_under(U_h1, prompts, base_logits)
                kl_h0 += kl_under(U_h0, prompts, base_logits)
                # Fresh draws per batch, shared within it. Projection ablation depends only on
                # span(U), so a Haar subspace is the dimension- and spectrum-matched control.
                rs = [haar_basis(d, k, rng) for _ in range(self.n_controls)]
                ps = [np.linalg.qr(P @ rng.standard_normal((P.shape[1], k)))[0] for _ in range(self.n_controls)]
                e_rand += [energy(U, S2) for U in rs]
                e_spanp += [energy(U, S2) for U in ps]
                kl_rand.append([kl_under(U, prompts, base_logits) for U in rs])
                kl_spanp.append([kl_under(U, prompts, base_logits) for U in ps])
                kl_pca += kl_under(U_pca, prompts, base_logits)
        finally:
            handle.remove()

        kl_h1, kl_h0 = np.array(kl_h1), np.array(kl_h0)
        kl_rand = np.concatenate([np.array(r) for r in kl_rand], axis=1)
        kl_spanp = np.concatenate([np.array(r) for r in kl_spanp], axis=1)
        kl_pca = np.array(kl_pca)[None, :]
        return {
            'layer': layer,
            'n_eval': len(kl_h1),
            'n_controls': self.n_controls,
            'seconds': time.time() - t0,
            'h1_eigvals': [float(v) for v in h1_eigvals],
            'summary': summarize(kl_h1, kl_h0, kl_rand, self.n_boot, self.seed),
            'vs_span_p': versus(kl_h1, kl_spanp, self.n_boot, self.seed),
            'vs_pca_top': versus(kl_h1, kl_pca, self.n_boot, self.seed),
            'energy_removed': {'h1': energy(U_h1.astype(np.float64), S2), 'h0': energy(U_h0.astype(np.float64), S2),
                               'pca_top': energy(U_pca, S2), 'haar_mean': float(np.mean(e_rand)),
                               'span_p_mean': float(np.mean(e_spanp))},
            'kl_pca_top': kl_pca[0].tolist(),
            'kl_span_p_mean': kl_spanp.mean(0).tolist(),
            'kl_h1': kl_h1.tolist(),
            'kl_h0': kl_h0.tolist(),
            'kl_rand_mean': kl_rand.mean(0).tolist(),
            'kl_rand_max': kl_rand.max(0).tolist(),
        }


@dataclass
class Dormancy(HeldOutConfig):
    """Dormancy table (tab:illusion): rules out the subspace-patching illusion for H^1.

    1. Dormancy: on-distribution projection spread of H^1 vs H^0 and
       Haar-random subspaces (a dormant direction has near-constant projections).
    2. KL under projection ablation of the layer output onto S-perp at the last
       prompt position, H^1 vs dimension-matched Haar-random subspaces.
    3. Behaviour: top-1 next-token flip rate and target-entity recall in
       50-token greedy continuations of the bare prompt, H^1 vs random ablation.
    """

    quantize: Literal['none', '4bit'] = 'none'
    output_dir: str = 'outputs/dormancy_illusion'
    n_eval: int = 100
    n_random: int = 5
    n_gen_random: int = 2
    """Random controls that also run generation (costly)."""

    def run(self):
        out_path = results_path(self.output_dir, self.model)
        rng = np.random.default_rng(self.seed)
        facts, train_ids, test_ids = load_heldout_facts(self)
        train_a, train_b = make_train_pairs(facts, train_ids)

        model, tokenizer = load_causal_model(self.model, quantize=self.quantize)
        device = next(model.parameters()).device
        layer = int(self.layer_frac * n_layers(model))
        extract = partial(pooled_hidden_states, model, tokenizer, layer=layer, batch_size=self.batch_size,
                          max_length=64, fp32_pool=True)
        H_a, H_b = extract(train_a), extract(train_b)
        U_h1, h1_eigvals = fit_sheaf_h1_basis(H_a, H_b, self.k, self.edge_dim)
        U_h0 = fit_h0_pca(H_a, H_b, k=self.k, edge_dim=self.edge_dim)
        randoms = [haar_basis(model.config.hidden_size, self.k, rng) for _ in range(self.n_random)]

        H_test = extract([facts[i]['expressions'][0] for i in test_ids][:400]).numpy()
        dormancy = {
            'H1_spread': projection_spread(H_test, U_h1),
            'H0_spread': projection_spread(H_test, U_h0),
            'random_spread_mean': float(np.mean([projection_spread(H_test, R) for R in randoms])),
        }
        dormancy['H1_over_random'] = dormancy['H1_spread'] / max(dormancy['random_spread_mean'], 1e-12)

        ablator = ProjectionAblator(U_h1, device, next(model.parameters()).dtype)
        handle = get_layers(model)[layer].register_forward_hook(ablator)
        try:
            per_prompt = [self._prompt(model, tokenizer, ablator, facts[i], U_h1, randoms, device)
                          for i in test_ids[:self.n_eval]]
        finally:
            ablator.enabled = False
            handle.remove()

        def mean(key, rows=per_prompt):
            return float(np.mean([r[key] for r in rows]))

        ratios = np.array([r['kl_ratio'] for r in per_prompt])
        summary = {
            'kl_ratio_mean': float(ratios.mean()),
            'kl_ratio_median': float(np.median(ratios)),
            'flip_rate_h1': mean('flip_h1'),
            'flip_rate_random': mean('flip_rand_mean'),
            'recall_base': mean('recall_base'),
            'recall_h1': mean('recall_h1'),
            'recall_random': mean('recall_rand_mean', [r for r in per_prompt if r['recall_rand_mean'] is not None]),
        }
        write_json(out_path, {
            'model': self.model, 'layer': layer, 'k': self.k, 'edge_dim': self.edge_dim,
            'n_eval': len(per_prompt), 'n_random': self.n_random, 'seed': self.seed,
            'h1_eigvals': [float(v) for v in h1_eigvals],
            'dormancy': dormancy, 'summary': summary, 'per_prompt': per_prompt,
        })

    def _prompt(self, model, tokenizer, ablator, fact, U_h1, randoms, device):
        """Logit KL, top-1 flip and entity recall of one bare prompt under H^1 and random ablation."""
        prompt = fact['prompt']  # bare prompt: the fact sentence would hand the model the answer
        entity = fact['entity'].lower()

        def ablated(U, generate):
            ablator.U = torch.from_numpy(U).to(device=device, dtype=ablator.U.dtype)
            ablator.enabled = True
            logits = last_token_logits(model, tokenizer, prompt, device)
            recalled = int(entity in greedy_generation(model, tokenizer, prompt, device).lower()) if generate else None
            return logits, recalled

        ablator.enabled = False
        base_logits = last_token_logits(model, tokenizer, prompt, device)
        recall_base = int(entity in greedy_generation(model, tokenizer, prompt, device).lower())
        h1_logits, recall_h1 = ablated(U_h1, True)
        kl_rand, flip_rand, recall_rand = [], [], []
        for r_i, R in enumerate(randoms):
            r_logits, recalled = ablated(R, r_i < self.n_gen_random)
            kl_rand.append(kl_divergence(base_logits, r_logits))
            flip_rand.append(int(base_logits.argmax() != r_logits.argmax()))
            if recalled is not None:
                recall_rand.append(recalled)

        row = {'case_id': fact['case_id'], 'kl_h1': kl_divergence(base_logits, h1_logits),
               'flip_h1': int(base_logits.argmax() != h1_logits.argmax()),
               'recall_base': recall_base, 'recall_h1': recall_h1,
               'kl_rand_mean': float(np.mean(kl_rand)), 'flip_rand_mean': float(np.mean(flip_rand)),
               'recall_rand_mean': float(np.mean(recall_rand)) if recall_rand else None}
        row['kl_ratio'] = row['kl_h1'] / max(row['kl_rand_mean'], 1e-12)
        return row


EXPERIMENTS = {
    'coordinates': Coordinates,
    'eigenbasis': Eigenbasis,
    'dormancy': Dormancy,
}

if __name__ == '__main__':
    tyro.extras.subcommand_cli_from_dict(EXPERIMENTS, description=__doc__, use_underscores=True).run()
