"""
Held-out CounterFact retrieval shared by the LEACE, restriction-map, cycle,
cross-dataset, sensitivity, baseline, and dormancy experiments.

Includes:

* ``HeldOutData`` / ``HeldOutConfig``: the common experiment flags.
* ``HeldOutRetrieval``: the seeded 70/30 fact split with centroid / query
  halves, and the top-1 scorer with optional same-relation hard negatives.
* ``encode``: pooled activations of several prompt lists from one model load.
* ``load_heldout_facts``: the CounterFact facts and the seeded fact split.
* ``make_train_pairs`` and ``fit_random_bases``: the training paraphrase
  pairs and the Random-k control bases.
* ``cosine_sim``, ``retrieval_top1`` and ``retrieval_with_basis`` for
  centroid lookup through an optional (d, k) basis.
* ``fit_sheaf_h1_basis`` for the top-k (H^1) eigenbasis used by the dormancy
  and cross-lingual probes.
"""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Literal, Optional, Sequence, Tuple

import numpy as np
import torch

from sheafint.core.h0 import edge_laplacian_eig, fit_h0_with_projection, joint_pca_projection

from .datasets import (
    build_centroid_query_split,
    hard_restrict_by_relation,
    load_counterfact_with_paraphrases,
    split_facts_train_test,
)
from .model_io import load_causal_model, n_layers, pooled_hidden_states


@dataclass
class HeldOutData:
    model: str
    """Hugging Face model id."""
    n_facts: int = 500
    """CounterFact facts to sample (each with at least 4 expressions)."""
    train_frac: float = 0.70
    """Fraction of facts that fit the bases; the rest are held out."""
    layer_frac: float = 0.67
    """Extraction layer as a fraction of model depth."""
    edge_dim: int = 128
    """Joint-PCA restriction-map dimension."""
    quantize: Literal['none', '4bit'] = '4bit'
    """4-bit NF4 or unquantized fp16."""
    output_dir: str = 'outputs'
    seed: int = 42
    batch_size: int = 8


@dataclass
class HeldOutConfig(HeldOutData):
    k: int = 20
    """Subspace dimension."""


def encode(cfg, text_lists: Sequence[List[str]], **pool) -> Tuple[List[torch.Tensor], int, int]:
    """Pool every prompt list at ``cfg.layer_frac`` depth from one load of ``cfg.model``.

    The model is freed before returning ``(activations, layer, hidden_size)``;
    ``pool`` is passed to :func:`model_io.pooled_hidden_states`.
    """
    model, tokenizer = load_causal_model(cfg.model, quantize=cfg.quantize)
    layer = int(cfg.layer_frac * n_layers(model))
    hidden = model.config.hidden_size
    out = [pooled_hidden_states(model, tokenizer, texts, layer, batch_size=cfg.batch_size, **pool)
           for texts in text_lists]
    del model
    torch.cuda.empty_cache()
    return out, layer, hidden


def load_heldout_facts(cfg, min_expressions: int = 4) -> Tuple[List[dict], List[int], List[int]]:
    """CounterFact facts with the seeded train / held-out fact split."""
    facts = load_counterfact_with_paraphrases(n_facts=cfg.n_facts, min_expressions=min_expressions, seed=cfg.seed)
    train_ids, test_ids = split_facts_train_test(facts, train_frac=cfg.train_frac, seed=cfg.seed)
    return facts, train_ids, test_ids


@dataclass
class HeldOutRetrieval:
    """Seeded CounterFact fact split with per-fact centroid / query halves."""

    facts: List[dict]
    train_ids: List[int]
    test_ids: List[int]
    split: Dict[str, list]
    hard_restrict: List[List[int]]

    @classmethod
    def from_args(cls, args, min_expressions: int = 4) -> 'HeldOutRetrieval':
        facts, train_ids, test_ids = load_heldout_facts(args, min_expressions)
        split = build_centroid_query_split(facts, test_ids, seed=args.seed)
        hard = hard_restrict_by_relation(facts, test_ids, split['query_relation_id'])
        return cls(facts, train_ids, test_ids, split, hard)

    @property
    def centroid_prompts(self) -> List[str]:
        return self.split['centroid_prompts']

    @property
    def query_prompts(self) -> List[str]:
        return self.split['query_prompts']

    def score(
        self,
        queries: np.ndarray,
        centroids: np.ndarray,
        basis: Optional[np.ndarray] = None,
        hard: bool = True,
        rows: Optional[np.ndarray] = None,
    ) -> float:
        """Top-1 accuracy; ``hard`` restricts candidates to same-relation test facts.

        ``rows`` scores only those queries (e.g. a bootstrap resample).
        """
        query_ids = self.split['query_fact_id']
        restrict = self.hard_restrict if hard else None
        if rows is not None:
            queries, query_ids = queries[rows], query_ids[rows]
            restrict = [restrict[j] for j in rows] if hard else None
        return retrieval_with_basis(
            queries, centroids, basis, self.split['centroid_fact_id'], query_ids, restrict_to=restrict,
        )

    def run_info(self, cfg, layer: int, hidden: int) -> Dict:
        """Run metadata shared by every held-out payload."""
        return {
            'model': cfg.model,
            'hidden_dim': hidden,
            'layer': layer,
            'n_train_facts': len(self.train_ids),
            'n_test_facts': len(self.test_ids),
            'edge_dim': cfg.edge_dim,
            'seed': cfg.seed,
        }


def make_train_pairs(facts: list, train_ids: Iterable[int], chain: int = 3) -> Tuple[List[str], List[str]]:
    """Consecutive paraphrase pairs (p0, p1), (p1, p2), ... up to ``chain`` per fact."""
    a: List[str] = []
    b: List[str] = []
    for i in train_ids:
        exprs = facts[i]['expressions']
        for j in range(min(chain, len(exprs) - 1)):
            a.append(exprs[j])
            b.append(exprs[j + 1])
    return a, b


def fit_random_bases(d: int, k: int, n_trials: int, seed: int) -> List[np.ndarray]:
    """``n_trials`` random orthonormal (d, k) bases for the Random-k control."""
    rng = np.random.default_rng(seed + 7)
    out: List[np.ndarray] = []
    for _ in range(n_trials):
        r, _ = np.linalg.qr(rng.standard_normal((d, k)).astype(np.float32))
        out.append(r)
    return out


def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Pairwise cosine similarity between row vectors.

    Both arrays are L2-normalised row-wise before the inner product so the
    output is the standard cosine kernel; constants in the denominator avoid
    division by zero on rare all-zero rows.
    """
    a_n = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-10)
    b_n = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-10)
    return a_n @ b_n.T


def retrieval_top1(
    centroids: np.ndarray,
    queries: np.ndarray,
    query_labels: np.ndarray,
    restrict_to: Optional[Sequence[Sequence[int]]] = None,
) -> float:
    """
    Top-1 retrieval accuracy. ``query_labels`` indexes ``centroids``. If
    ``restrict_to`` is given, the candidate set per query is the listed
    centroid indices; queries with at most one candidate are skipped.
    """
    sims = cosine_sim(queries, centroids)
    if restrict_to is None:
        preds = np.argmax(sims, axis=1)
        return float((preds == query_labels).mean())

    correct = 0
    n_eval = 0
    for q in range(len(query_labels)):
        cands = restrict_to[q]
        if len(cands) <= 1:
            continue
        n_eval += 1
        pred_local = int(np.argmax(sims[q, cands]))
        if cands[pred_local] == query_labels[q]:
            correct += 1
    return float(correct / max(n_eval, 1))


def retrieval_with_basis(
    H_query: np.ndarray,
    H_centroid_pool: np.ndarray,
    basis: Optional[np.ndarray],
    centroid_fact_ids: np.ndarray,
    query_fact_ids: np.ndarray,
    restrict_to: Optional[Sequence[Sequence[int]]] = None,
) -> float:
    """
    Project queries and centroid-pool through ``basis`` (or use raw
    activations if basis is None), aggregate centroids per fact id, and
    evaluate top-1 retrieval. ``restrict_to`` takes lists of fact ids and
    is converted to centroid indices internally.
    """
    if basis is not None:
        Q = H_query @ basis
        C = H_centroid_pool @ basis
    else:
        Q = H_query
        C = H_centroid_pool

    unique_facts = np.unique(centroid_fact_ids)
    centroids = np.stack([
        C[centroid_fact_ids == f].mean(axis=0) for f in unique_facts
    ])
    fact_to_idx = {int(f): i for i, f in enumerate(unique_facts)}
    query_idx = np.array([fact_to_idx[int(f)] for f in query_fact_ids])

    restrict_idx = None
    if restrict_to is not None:
        restrict_idx = [
            [fact_to_idx[int(f)] for f in cand_facts if int(f) in fact_to_idx]
            for cand_facts in restrict_to
        ]
    return retrieval_top1(centroids, Q, query_idx, restrict_idx)


def fit_sheaf_h1_basis(pairs_a, pairs_b, k: int = 20, edge_dim: int = 128):
    """Top-k counterpart of :func:`sheafint.fit_h0_pca`; also returns the top-k eigenvalues."""
    P = joint_pca_projection(pairs_a, pairs_b, edge_dim)
    eigvals, _ = edge_laplacian_eig(pairs_a, pairs_b, P)
    return fit_h0_with_projection(pairs_a, pairs_b, P, k=k, top=True), eigvals[-k:]
