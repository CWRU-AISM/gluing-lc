# Smoke tests verifying the public API loads and a tiny sheaf fits cleanly.

import numpy as np
import torch
from sheafint import (
    ParaphraseContext,
    PromptVariantContext,
    ScalableSheaf,
    create_standard_cover,
)
from sheafint.baselines import (
    cka_consistency_energy,
    cosine_consistency_energy,
)
from sheafint.steering import joint_pca, sheaf_laplacian_decomposition


def test_paraphrase_context_loads():
    ctx = ParaphraseContext()
    assert len(ctx) > 0
    a, b = ctx.get_all_texts()
    assert len(a) == len(b)


def test_prompt_variant_context_pairs():
    ctx = PromptVariantContext()
    assert len(ctx) > 0


def test_standard_cover_builds():
    cover = create_standard_cover()
    cover_data = cover.build_cover()
    assert cover_data['nodes']
    assert cover_data['pair_overlaps']


def test_scalable_sheaf_fits_random_features():
    n_samples = 32
    hidden = 16
    rng = np.random.default_rng(0)

    feat_a = torch.tensor(rng.standard_normal((n_samples, hidden)).astype('float32'))
    noise = torch.tensor((rng.standard_normal((n_samples, hidden)) * 0.05).astype('float32'))
    feat_b = feat_a + noise

    sheaf = ScalableSheaf(edge_dim=8, face_dim=4, svd_rank=8, device='cpu')
    sheaf.fit(
        {'ctx_a': feat_a, 'ctx_b': feat_b},
        {('ctx_a', 'ctx_b'): torch.arange(n_samples)},
    )
    metrics = sheaf.evaluate(
        {'ctx_a': feat_a, 'ctx_b': feat_b},
        {('ctx_a', 'ctx_b'): torch.arange(n_samples)},
    )
    assert metrics.consistency_energy >= 0
    assert metrics.exactness_error < 1e-3


def test_baseline_methods_run():
    rng = np.random.default_rng(1)
    a = torch.tensor(rng.standard_normal((20, 8)).astype('float32'))
    b = a + 0.01 * torch.tensor(rng.standard_normal((20, 8)).astype('float32'))
    assert cosine_consistency_energy(a, b) < 1.0
    assert cka_consistency_energy(a, b) < 1.0


def test_sheaf_laplacian_decomposition_runs():
    rng = np.random.default_rng(2)
    s_left = rng.standard_normal((30, 12)).astype('float32')
    s_right = s_left + 0.05 * rng.standard_normal((30, 12)).astype('float32')
    background = rng.standard_normal((50, 12)).astype('float32')

    projection, _ = joint_pca(s_left, s_right, edge_dim=6)
    decomp = sheaf_laplacian_decomposition(s_left, s_right, background, projection, k=4)
    assert decomp['h0_vecs'].shape == (12, 4)
    assert decomp['h1_vecs'].shape == (12, 4)
