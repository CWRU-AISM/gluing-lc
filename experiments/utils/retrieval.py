"""
Retrieval helpers used by LEACE/restriction-map/cycle/cross-dataset experiments.

Includes:

* ``cosine_sim`` and ``retrieval_top1`` for centroid lookup.
* ``retrieval_with_basis`` for projecting through a (d, k) basis before
  scoring against per-fact centroids.
* ``pooled_hidden_states_variant`` for choosing among mean / last-token /
  first-token pooling.
"""

from typing import List, Optional, Sequence

import numpy as np
import torch


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
        sub_sims = sims[q, cands]
        pred_local = int(np.argmax(sub_sims))
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

    if restrict_to is not None:
        restrict_idx: Optional[List[List[int]]] = []
        for cand_facts in restrict_to:
            restrict_idx.append([
                fact_to_idx[int(f)] for f in cand_facts if int(f) in fact_to_idx
            ])
    else:
        restrict_idx = None

    return retrieval_top1(centroids, Q, query_idx, restrict_idx)


@torch.no_grad()
def pooled_hidden_states_variant(
    model,
    tokenizer,
    prompts: List[str],
    layer: int,
    pooling: str = 'mean',
    batch_size: int = 8,
    max_length: int = 64,
) -> torch.Tensor:
    """
    Hidden states pooled according to ``pooling`` in {'mean','last','first'}.
    Returns a stacked CPU float tensor.
    """
    device = next(model.parameters()).device
    out: List[torch.Tensor] = []
    for i in range(0, len(prompts), batch_size):
        chunk = prompts[i:i + batch_size]
        enc = tokenizer(
            chunk, return_tensors='pt', padding=True,
            truncation=True, max_length=max_length,
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        h = model(**enc, output_hidden_states=True).hidden_states[layer]
        mask = enc['attention_mask']
        if pooling == 'mean':
            m = mask.unsqueeze(-1).float()
            pooled = (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)
        elif pooling == 'last':
            lengths = mask.sum(dim=1).clamp(min=1) - 1
            idx = lengths.view(-1, 1, 1).expand(-1, 1, h.size(-1))
            pooled = h.gather(1, idx).squeeze(1)
        elif pooling == 'first':
            pooled = h[:, 0, :]
        else:
            raise ValueError(f"unknown pooling: {pooling}")
        out.append(pooled.float().cpu())
    return torch.cat(out, dim=0)
