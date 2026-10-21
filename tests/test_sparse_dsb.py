"""Checks of the IPF port and the model adapters. Run on a GPU node:

    sbatch --job-name=tests slurm/run.sbatch python -m pytest -x -q tests
"""
from __future__ import annotations

import os

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sparse_dsb.cache import flatten_trajectories, minibatch
from sparse_dsb.ipf import IPF, IPFConfig
from sparse_dsb.langevin import gaussian_gradient, record_init_langevin, record_langevin_seq
from sparse_dsb.models import ModelConfig, build_model, parameter_counts, patchify, unpatchify
from sparse_dsb.schedule import cumulative_time, make_gammas
from st.transformer import SparseLinear as MaskedSparseLinear
from z1t.components import SparseLinear as GatherSparseLinear

KEY = jax.random.PRNGKey(0)


def test_schedule_is_symmetric():
    gammas = make_gammas(50, 1e-5, 0.1)
    assert gammas.shape == (50,)
    np.testing.assert_allclose(gammas, gammas[::-1])
    assert gammas[0] == pytest.approx(1e-5) and gammas[24] == pytest.approx(0.1)
    time = cumulative_time(gammas)
    assert time[-1] == pytest.approx(gammas.sum())


def test_init_langevin_targets_match_definition():
    gammas = jnp.asarray(make_gammas(10, 1e-3, 1e-2))
    time = jnp.asarray(cumulative_time(np.asarray(gammas)))
    x0 = jax.random.normal(KEY, (4, 3, 8, 8))
    mean, var = jnp.float32(0.4), jnp.float32(0.5)
    x_tot, out, steps = record_init_langevin(KEY, x0, gammas, time, mean, var)
    assert x_tot.shape == (10, 4, 3, 8, 8) and out.shape == x_tot.shape and steps.shape == (10,)
    previous = jnp.concatenate([x0[None], x_tot[:-1]])
    t_old = previous + gammas[:, None, None, None, None] * gaussian_gradient(previous, mean, var)
    t_new = x_tot + gammas[:, None, None, None, None] * gaussian_gradient(x_tot, mean, var)
    np.testing.assert_allclose(np.asarray(out), np.asarray(t_old - t_new), atol=1e-6)
    noise = (x_tot - t_old) / jnp.sqrt(2 * gammas)[:, None, None, None, None]
    assert abs(float(noise.std()) - 1.0) < 0.1


def test_langevin_seq_targets_and_last_step():
    gammas = jnp.asarray(make_gammas(6, 1e-3, 1e-2))
    time = jnp.asarray(cumulative_time(np.asarray(gammas)))
    x0 = jax.random.normal(KEY, (5, 2))

    def net(xs, ts):
        return 0.9 * xs + ts[:, None]

    x_tot, out, _ = record_langevin_seq(KEY, net, x0, gammas, time, mean_match=True, sample=True)
    previous = jnp.concatenate([x0[None], x_tot[:-1]])
    t_old = 0.9 * previous + time[:, None, None]
    t_new = 0.9 * x_tot + time[:, None, None]
    np.testing.assert_allclose(np.asarray(out), np.asarray(t_old - t_new), atol=1e-6)
    np.testing.assert_allclose(np.asarray(x_tot[-1]), np.asarray(t_old[-1]), atol=1e-6)

    x_tot2, out2, _ = record_langevin_seq(KEY, net, x0, gammas, time, mean_match=False, sample=False)
    previous = jnp.concatenate([x0[None], x_tot2[:-1]])
    t_old = previous + 0.9 * previous + time[:, None, None]
    t_new = x_tot2 + 0.9 * x_tot2 + time[:, None, None]
    np.testing.assert_allclose(np.asarray(out2), np.asarray(t_old - t_new), atol=1e-5)


def test_cache_flattening_and_minibatch():
    num_steps, batch = 3, 4
    x_tot = jnp.arange(num_steps * batch, dtype=jnp.float32).reshape(num_steps, batch, 1)
    time = jnp.asarray([0.1, 0.2, 0.3])
    cache = flatten_trajectories(x_tot, -x_tot, time)
    assert cache.xs.shape == (12, 1)
    np.testing.assert_allclose(np.asarray(cache.xs[:3, 0]), [0.0, 4.0, 8.0])
    np.testing.assert_allclose(np.asarray(cache.steps[:4]), [0.1, 0.2, 0.3, 0.1])
    permutation = jax.random.permutation(KEY, 12)
    seen = jnp.concatenate([minibatch(cache, permutation, j, 4)[0][:, 0] for j in range(3)])
    assert sorted(np.asarray(seen).tolist()) == sorted(np.asarray(cache.xs[:, 0]).tolist())


def test_patchify_roundtrip():
    x = jax.random.normal(KEY, (3, 16, 16))
    tokens = patchify(x, 4)
    assert tokens.shape == (16, 48)
    np.testing.assert_allclose(np.asarray(unpatchify(tokens, 3, 16, 16, 4)), np.asarray(x))


@pytest.mark.parametrize("family, sparse_linear", [("st", "mask"), ("z1t", "mask"), ("z1t", "gather")])
def test_model_identity_at_init_and_fan_in(family, sparse_linear):
    config = ModelConfig(family=family, image_size=16, patch_size=4, n_layers=2, n_embed=64, n_head=4,
                         linear_fan_in=16, time_embed_dim=16, sparse_linear=sparse_linear)
    model = build_model(config, KEY)
    x = jax.random.normal(KEY, (3, 16, 16))
    np.testing.assert_allclose(np.asarray(model(x, jnp.float32(0.3))), np.asarray(x))
    sparse = [n for n in jax.tree.leaves(model, is_leaf=lambda n: isinstance(n, (MaskedSparseLinear, GatherSparseLinear)))
              if isinstance(n, (MaskedSparseLinear, GatherSparseLinear))]
    assert sparse, "expected sparse linears with K < input width"
    expected = GatherSparseLinear if sparse_linear == "gather" else MaskedSparseLinear
    assert all(isinstance(layer, expected) for layer in sparse)
    for layer in sparse:
        if isinstance(layer, MaskedSparseLinear):
            assert bool(jnp.all(layer.mask.sum(axis=1) == 16))
        else:
            assert layer.k == 16 and all(len(set(row)) == 16 for row in np.asarray(layer.indices).tolist())
    stored, active = parameter_counts(model)
    assert active <= stored


@pytest.mark.parametrize("family", ["st", "z1t"])
def test_context_prefix_gives_full_receptive_field(family):
    config = ModelConfig(family=family, image_size=8, patch_size=4, n_layers=2, n_embed=32, n_head=4, time_embed_dim=16,
                         context_prefix=True, residual_output=False)
    model = randomise_unembed(build_model(config, KEY))
    x = jax.random.normal(KEY, (3, 8, 8))
    y = model(x, jnp.float32(0.2))
    x2 = x.at[:, 4:, 4:].add(1.0)                  # perturb the last patch
    y2 = model(x2, jnp.float32(0.2))
    assert float(jnp.abs(y2[:, :4, :4] - y[:, :4, :4]).max()) > 1e-6, "first patch should see the last patch"

    causal = randomise_unembed(build_model(ModelConfig(**{**config.__dict__, "context_prefix": False}), KEY))
    y, y2 = causal(x, jnp.float32(0.2)), causal(x2, jnp.float32(0.2))
    assert float(jnp.abs(y2[:, :4, :4] - y[:, :4, :4]).max()) < 1e-6, "without the prefix the first patch is blind"


def randomise_unembed(model):
    """Replace the zero-initialised output projection by a random one of the same type."""
    layer = model.unembed
    key = jax.random.PRNGKey(3)
    if isinstance(layer, MaskedSparseLinear):
        inner = eqx.nn.Linear(layer.linear.in_features, layer.linear.out_features, key=key)
        layer = eqx.tree_at(lambda l: l.linear, layer, inner)
    elif isinstance(layer, GatherSparseLinear):
        layer = eqx.tree_at(lambda l: l.weight, layer, jax.random.normal(key, layer.weight.shape))
    else:
        layer = eqx.nn.Linear(layer.in_features, layer.out_features, key=key)
    return eqx.tree_at(lambda m: m.unembed, model, layer)


@pytest.mark.parametrize("family", ["st", "z1t"])
def test_ipf_smoke(tmp_path, family):
    if jax.default_backend() != "gpu":
        pytest.skip("IPF smoke test runs on GPU only")
    images = np.asarray(jax.random.randint(KEY, (64, 8, 8, 3), 0, 256), dtype=np.uint8)
    ipf_config = IPFConfig(num_steps=6, n_ipf=2, num_iter=12, batch_size=16, cache_npar=8, cache_refresh_stride=6,
                           log_stride=3, gif_stride=6, plot_npar=4, num_cache_batches=2, devices=1)
    model_config = ModelConfig(family=family, image_size=8, patch_size=4, n_layers=2, n_embed=32, n_head=4,
                               linear_fan_in=8, time_embed_dim=16)
    ipf = IPF(ipf_config, model_config, images, str(tmp_path))
    ipf.train()
    names = sorted(os.listdir(tmp_path / "checkpoints"))
    assert "net_b_1_12.eqx" in names and "sample_net_f_2_12.eqx" in names
    assert (tmp_path / "im" / "b_2_12" / "im_grid_final.png").exists()
    rows = np.genfromtxt(tmp_path / "logs" / "metrics.csv", delimiter=",", names=True, dtype=None, encoding="utf-8")
    assert rows.size == 2 * 2 * 4 and np.all(np.isfinite(rows["loss"]))
