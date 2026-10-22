"""The toy distributions, the score-network port and a tiny toy IPF.

    sbatch --job-name=toytest slurm/run.sbatch python -m pytest -x -q -rA tests/test_toy.py

The score network is compared with the reference ScoreNetwork (bridge/models/basic, CPU torch of the
`validation` group) on identical weights; the IPF smoke needs a GPU.
"""
from __future__ import annotations

import importlib
import os
import sys
import types

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sparse_dsb.ipf import IPF, IPFConfig
from sparse_dsb.toy import (DISTRIBUTIONS, ScoreNetwork, ToyModelConfig, energy_distance, fairseq_embedding, toy_points,
                            toy_prior)
from st.transformer import SparseLinear as MaskedSparseLinear
from z1t.components import SparseLinear as GatherSparseLinear

REFERENCE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "diffusion_schrodinger_bridge")
KEY = jax.random.PRNGKey(0)


@pytest.mark.parametrize("distribution", DISTRIBUTIONS)
def test_points_follow_the_reference_recipe(distribution):
    points = toy_points(distribution, 10000, 0)
    assert points.shape == (10000, 2) and points.dtype == np.float32
    if distribution in ("moon", "scurve"):
        np.testing.assert_allclose(points.mean(), 0.0, atol=1e-3)
        np.testing.assert_allclose(points.std(ddof=1), 7.0, rtol=1e-4)
    elif distribution == "circle":
        radii = np.linalg.norm(points, axis=1)
        outer, inner = np.abs(radii - 10.0) < 1e-3, np.abs(radii - 5.0) < 1e-3
        assert np.all(outer | inner) and outer.sum() == 5000
    else:
        assert abs(points[:5000, 0].mean() + 7.0) < 0.1 and abs(points[5000:, 0].mean() - 7.0) < 0.1
        assert abs(points[:, 1].std() - 1.0) < 0.05
    assert np.array_equal(toy_points(distribution, 10000, 0), points)
    assert not np.array_equal(toy_points(distribution, 10000, 1), points)


def test_prior_is_the_first_hundred_points():
    points = toy_points("mixture", 1000, 0)
    mean, var = toy_prior(points)
    np.testing.assert_allclose(mean, points[:100].mean(axis=0), rtol=1e-5)
    np.testing.assert_allclose(var, points[:100].astype(np.float64).var(ddof=1), rtol=1e-6)
    assert mean[0] < -6.0


def test_energy_distance():
    generator = np.random.default_rng(0)
    a, b = generator.standard_normal((500, 2)), generator.standard_normal((500, 2))
    assert abs(energy_distance(a, a)) < 1e-9
    assert energy_distance(a, b) < 0.05
    assert energy_distance(a, b + 5.0) > 5.0


@pytest.mark.parametrize("sparse_linear, fan_in, expected", [
    ("mask", 32, {(2, 16): 2, (16, 32): 16, (64, 128): 32, (128, 128): 32, (128, 2): 32, (16, 16): 16}),
    ("mask", None, {(2, 16): 2, (16, 32): 16, (64, 128): 64, (128, 128): 128, (128, 2): 128, (16, 16): 16}),
    ("gather", 32, {(2, 16): 2, (16, 32): 16, (64, 128): 32, (128, 128): 32, (128, 2): 32, (16, 16): 16}),
])
def test_sparse_layers_have_the_clamped_fan_in(sparse_linear, fan_in, expected):
    model = ScoreNetwork(ToyModelConfig(sparse_linear=sparse_linear, linear_fan_in=fan_in), KEY)
    layers = [node for node in jax.tree.leaves(model, is_leaf=lambda n: isinstance(n, (MaskedSparseLinear, GatherSparseLinear)))
              if isinstance(node, (MaskedSparseLinear, GatherSparseLinear))]
    assert len(layers) == 7
    for layer in layers:
        if isinstance(layer, MaskedSparseLinear):
            ins, outs = layer.linear.in_features, layer.linear.out_features
            assert np.all(np.asarray(layer.mask.sum(axis=1)) == expected[(ins, outs)])
        else:
            assert layer.indices.shape == (layer.out_features, expected[(layer.in_features, layer.out_features)])
    out = jax.vmap(model)(jnp.ones((3, 2)), jnp.full((3,), 0.1))
    assert out.shape == (3, 2) and np.all(np.isfinite(np.asarray(out)))


def load_reference_basic():
    torch = pytest.importorskip("torch")
    if REFERENCE not in sys.path:
        sys.path.insert(0, REFERENCE)
    if "bridge.data" not in sys.modules:
        importlib.import_module("bridge")
        package = types.ModuleType("bridge.data")
        package.__path__ = [os.path.join(REFERENCE, "bridge", "data")]
        sys.modules["bridge.data"] = package
    return torch, importlib.import_module("bridge.models.basic.basic"), importlib.import_module("bridge.models.basic.time_embedding")


def test_fairseq_embedding_matches_reference(check_close):
    torch, _, time_embedding = load_reference_basic()
    times = np.array([0.0, 0.01, 0.13, 0.2, 1.0], np.float32)
    port = np.stack([np.asarray(fairseq_embedding(t, 16)) for t in times])
    reference = time_embedding.get_timestep_embedding(torch.tensor(times)[:, None], 16).numpy()
    check_close("embedding", port, reference, rtol=1e-6, atol=1e-6)


def test_score_network_matches_reference_on_shared_weights(check_close):
    torch, basic, _ = load_reference_basic()
    torch.manual_seed(0)
    reference = basic.ScoreNetwork()
    model = ScoreNetwork(ToyModelConfig(), KEY)
    for name in ("net", "t_encoder", "x_encoder"):
        for index, layer in enumerate(getattr(reference, name).layers):
            weight, bias = layer.weight.detach().numpy(), layer.bias.detach().numpy()
            model = eqx.tree_at(lambda m, name=name, index=index: getattr(m, name).layers[index].weight, model, jnp.asarray(weight))
            model = eqx.tree_at(lambda m, name=name, index=index: getattr(m, name).layers[index].bias, model, jnp.asarray(bias))
    assert sum(parameter.numel() for parameter in reference.parameters()) == 26498
    assert sum(int(leaf.size) for leaf in jax.tree.leaves(eqx.filter(model, eqx.is_inexact_array))) == 26498
    generator = np.random.default_rng(0)
    x = generator.standard_normal((64, 2)).astype(np.float32) * 7
    t = generator.uniform(0, 0.2, 64).astype(np.float32)
    with jax.default_matmul_precision("highest"):
        port = np.asarray(jax.vmap(model)(jnp.asarray(x), jnp.asarray(t)))
    with torch.no_grad():
        expected = reference(torch.tensor(x), torch.tensor(t)[:, None]).numpy()
    check_close("score_network", port, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(jax.default_backend() != "gpu", reason="training runs on GPU")
@pytest.mark.parametrize("sparse_linear, fan_in", [("dense", None), ("mask", 8)])
def test_toy_ipf_smoke(tmp_path, sparse_linear, fan_in):
    points = toy_points("moon", 512, 0)
    config = IPFConfig(num_steps=4, gamma_min=0.01, gamma_max=0.01, mean_match=False, adaptive_mean=False, final_adaptive=True,
                       n_ipf=2, num_iter=20, batch_size=64, lr=1e-3, grad_clipping=False, ema=False, use_prev_net=False,
                       cache_npar=256, cache_refresh_stride=10, random_flip=False, log_stride=5, gif_stride=20, plot_npar=128,
                       devices=1)
    ipf = IPF(config, ToyModelConfig(sparse_linear=sparse_linear, linear_fan_in=fan_in), points, str(tmp_path))
    assert ipf.shape == (2,) and ipf.info["kind"] == "toy"
    np.testing.assert_allclose(np.asarray(ipf.mean_final), points[:100].mean(axis=0), rtol=1e-5)
    ipf.train()
    metrics = np.genfromtxt(tmp_path / "logs" / "metrics.csv", delimiter=",", names=True, dtype=None, encoding="utf-8")
    assert metrics.size == 16 and np.all(np.isfinite(metrics["loss"]))
    toy_metrics = np.genfromtxt(tmp_path / "logs" / "toy_metrics.csv", delimiter=",", names=True, dtype=None, encoding="utf-8")
    tags = [f"{row['fb']}_{row['n']}_{row['i']}" for row in toy_metrics]
    assert tags == ["f_0_0", "b_1_20", "f_1_20", "b_2_20", "f_2_20"]
    assert np.all(np.isfinite(toy_metrics["energy_to_data"])) and np.all(toy_metrics["energy_to_prior"] >= -1e-6)
    for name in ("im/data.png", "im/b_2_20/samples.png", "im/b_2_20/trajectories.png", "im/f_2_20/samples.png", "checkpoints/net_b_2_20.eqx"):
        assert (tmp_path / name).exists(), name
    assert not (tmp_path / "checkpoints" / "sample_net_b_2_20.eqx").exists()
