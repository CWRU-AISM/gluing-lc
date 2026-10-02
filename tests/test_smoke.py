# Smoke tests: the public API loads, a tiny sheaf fits, and the model-free
# topology and statistics helpers reproduce known values.

import dataclasses
import importlib

import numpy as np
import pytest
import torch
import tyro
from sheafint import ScalableSheaf
from sheafint.steering import joint_pca, sheaf_laplacian_decomposition


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
    assert sheaf.fitted


def test_sheaf_laplacian_decomposition_runs():
    rng = np.random.default_rng(2)
    s_left = rng.standard_normal((30, 12)).astype('float32')
    s_right = s_left + 0.05 * rng.standard_normal((30, 12)).astype('float32')
    background = rng.standard_normal((50, 12)).astype('float32')

    projection, _ = joint_pca(s_left, s_right, edge_dim=6)
    decomp = sheaf_laplacian_decomposition(s_left, s_right, background, projection, k=4)
    assert decomp['h0_vecs'].shape == (12, 4)
    assert decomp['h1_vecs'].shape == (12, 4)


def test_spearman_exact_reproduces_fragility_stat():
    from utils.statistics import spearman_exact
    mass = [0.0001, 0.0001, 0.001, 0.033, 0.036, 0.079]
    preservation = [39.4, 34.3, 36.0, 27.5, 9.1, 4.2]
    rho, p = spearman_exact(mass, [-v for v in preservation])
    assert abs(rho - 0.899) < 1e-3 and abs(p - 10 / 720) < 1e-9


def test_projection_ablator_removes_haar_subspace():
    from utils.causal import ProjectionAblator, haar_basis, kl_divergence
    U = haar_basis(16, 4, np.random.default_rng(0))
    assert np.allclose(U.T @ U, np.eye(4))
    ablator = ProjectionAblator(U, 'cpu', torch.float64)
    ablator.enabled = True
    h = torch.randn(2, 5, 16, dtype=torch.float64)
    out = ablator(None, None, (h,))[0]
    assert torch.allclose(out @ ablator.U, torch.zeros(2, 5, 4, dtype=torch.float64), atol=1e-10)
    assert abs(kl_divergence(h[0, 0], h[0, 0])) < 1e-12


def test_hodge_complexes_have_expected_betti_numbers():
    from sheafint.core.hypergraph import simplicial_coboundaries
    from sheafint.data import build_facts
    from utils.hodge import betti, build_fact_complex, build_grid, hodge_2d

    facts = build_facts(n_relations=6, paraphrases_per_fact=5)
    for mode, expected in [('ring', (6, 66, 0)), ('triangulated', (6, 6, 240))]:
        H = build_fact_complex(facts, 5, mode, relation_rings=True)
        b = betti({n: i for i, n in enumerate(H.nodes)}, H.edges, H.faces)
        assert (b['b_0'], b['b_1'], b['b_2']) == expected
    for N, L in [(30, 2), (30, 4), (5, 8)]:
        node_idx, horiz, vert, _, _ = build_grid(N, L)
        assert betti(node_idx, horiz + vert, [])['b_1'] == N * (L - 1)
        node_idx, horiz, vert, diag, faces = build_grid(N, L, diagonals=True)
        b = betti(node_idx, horiz + vert + diag, faces)
        assert (b['b_1'], b['b_2']) == (0, 0)

    # The Kronecker-structured Hodge split equals the dense lifted pseudo-inverse.
    node_idx, horiz, vert, diag, faces = build_grid(4, 3, diagonals=True)
    d0, d1 = simplicial_coboundaries(node_idx, horiz + vert + diag, faces)
    k = 3
    a = np.random.default_rng(0).standard_normal((d0.shape[0], k))
    exact, _, coexact = hodge_2d(a, d0, d1)
    D0, D1 = np.kron(d0, np.eye(k)), np.kron(d1, np.eye(k))
    e1 = D0 @ np.linalg.lstsq(D0, a.reshape(-1), rcond=None)[0]
    c1 = D1.T @ np.linalg.lstsq(D1.T, a.reshape(-1) - e1, rcond=None)[0]
    assert np.allclose(e1, exact.reshape(-1)) and np.allclose(c1, coexact.reshape(-1))


@pytest.mark.parametrize('family', ['retrieval', 'topology', 'causal', 'steering'])
def test_entry_point_experiments_parse(family):
    module = importlib.import_module(family)
    for name, cls in module.EXPERIMENTS.items():
        assert dataclasses.is_dataclass(cls) and callable(cls.run) and cls.__doc__
        args = [name] + (['--model', 'gpt2'] if 'model' in {f.name for f in dataclasses.fields(cls)} else [])
        cfg = tyro.extras.subcommand_cli_from_dict(module.EXPERIMENTS, args=args, use_underscores=True)
        assert isinstance(cfg, cls)
