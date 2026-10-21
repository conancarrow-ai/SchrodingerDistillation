"""Data-parallel IPF over the devices of one node must reproduce the single-device run.

    sbatch --job-name=sharding --gres=gpu:rtx6000:2 slurm/run.sbatch python -m pytest -x -q -rA tests/test_sharding.py

Skips when fewer than two devices are visible. For each model family the tiny
synthetic IPF smoke of test_sparse_dsb.py runs once on every visible device and
once on device 0 with the same seed; the logged losses, the sample statistics and
the sample grids must agree up to the reduction order of float32 sums, the sparse
supports (masks or gather indices) must be identical, and a checkpoint of the
sharded run must hold a single copy and load into a single-device IPF. The EMA rate
is 0.5 so that the sampled grids reflect the trained networks.
"""
from __future__ import annotations

import dataclasses
import filecmp

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from PIL import Image

from sparse_dsb.ipf import IPF, IPFConfig
from sparse_dsb.models import ModelConfig
from st.transformer import SparseLinear as MaskedSparseLinear
from z1t.components import SparseLinear as GatherSparseLinear

pytestmark = pytest.mark.skipif(jax.default_backend() != "gpu" or jax.device_count() < 2,
                                reason="needs at least two GPUs")

TOLERANCE = 1e-3
SPARSE_KINDS = (MaskedSparseLinear, GatherSparseLinear)
IMAGES = np.random.default_rng(0).integers(0, 256, (64, 8, 8, 3), dtype=np.uint8)


def configs(family, sparse_linear, devices):
    ipf_config = IPFConfig(num_steps=6, n_ipf=2, num_iter=12, batch_size=16, cache_npar=8, cache_refresh_stride=6,
                           log_stride=3, gif_stride=6, plot_npar=8, num_cache_batches=2, devices=devices, ema_rate=0.5)
    model_config = ModelConfig(family=family, image_size=8, patch_size=4, n_layers=2, n_embed=32, n_head=4,
                               linear_fan_in=8, time_embed_dim=16, sparse_linear=sparse_linear)
    return ipf_config, model_config


def read_csv(path):
    return np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")


def run(out_dir, family, sparse_linear, devices):
    ipf = IPF(*configs(family, sparse_linear, devices), IMAGES, str(out_dir))
    ipf.train()
    return ipf, read_csv(out_dir / "logs" / "metrics.csv"), read_csv(out_dir / "logs" / "plot_logs.csv")


def sparse_layers(model):
    return [node for node in jax.tree.leaves(model, is_leaf=lambda n: isinstance(n, SPARSE_KINDS)) if isinstance(node, SPARSE_KINDS)]


def support(layer):
    return layer.mask if isinstance(layer, MaskedSparseLinear) else layer.indices


def relative_difference(a, b):
    return float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-12)))


def grid(path):
    return np.asarray(Image.open(path), dtype=np.int16)


def evaluate(net, xs, ts):
    return eqx.filter_jit(lambda module, x, t: jax.vmap(module)(x, t))(net, xs, ts)


@pytest.mark.parametrize("family, sparse_linear", [("st", "mask"), ("z1t", "mask"), ("z1t", "gather")])
def test_sharded_run_matches_single_device(tmp_path, record_property, family, sparse_linear):
    sharded, sharded_metrics, sharded_plots = run(tmp_path / "all", family, sparse_linear, devices=0)
    single, single_metrics, single_plots = run(tmp_path / "one", family, sparse_linear, devices=[0])
    assert len(sharded.devices) == jax.device_count() and len(single.devices) == 1
    label = f"{family}/{sparse_linear} on {jax.device_count()} devices"
    record_property("devices", jax.device_count())

    assert np.all(np.isfinite(sharded_metrics["loss"])) and sharded_metrics.size == single_metrics.size
    loss_difference = relative_difference(sharded_metrics["loss"], single_metrics["loss"])
    grad_difference = relative_difference(sharded_metrics["grad_norm"], single_metrics["grad_norm"])
    print(f"{label}: max relative difference loss {loss_difference:.2e}, grad_norm {grad_difference:.2e}")
    record_property("loss.max_rel_diff", loss_difference)
    record_property("grad_norm.max_rel_diff", grad_difference)
    assert loss_difference < TOLERANCE and grad_difference < TOLERANCE

    for column in ("init_var", "final_var", "mean_init", "mean_final"):
        difference = relative_difference(sharded_plots[column], single_plots[column])
        print(f"{label}: max relative difference {column} {difference:.2e}")
        record_property(f"{column}.max_rel_diff", difference)
        assert difference < TOLERANCE

    tags = [f"{row['fb']}_{row['n']}_{row['i']}" for row in sharded_plots]
    assert tags == ["f_0_0", "b_1_6", "b_1_12", "f_1_6", "f_1_12", "b_2_6", "b_2_12", "f_2_6", "f_2_12"]
    differing = 0.0
    for tag in tags:
        for name in ("im_grid_first.png", "im_grid_final.png"):
            a, b = grid(tmp_path / "all" / "im" / tag / name), grid(tmp_path / "one" / "im" / tag / name)
            assert a.shape == b.shape and np.abs(a - b).max() <= 1, (tag, name)
            differing = max(differing, float(np.mean(a != b)))
    print(f"{label}: at most {100 * differing:.3f}% of grid pixels differ by one level")
    record_property("grid_pixels.fraction_differing", differing)
    assert differing < 0.01

    for direction in ("f", "b"):
        leaves_sharded = jax.tree.leaves(eqx.filter(sharded.nets[direction], eqx.is_inexact_array))
        leaves_single = jax.tree.leaves(eqx.filter(single.nets[direction], eqx.is_inexact_array))
        assert len(leaves_sharded) == len(leaves_single)
        parameter_difference = max(float(np.max(np.abs(np.asarray(a) - np.asarray(b)))) for a, b in zip(leaves_sharded, leaves_single))
        print(f"{label}: max absolute difference of the trained {direction} parameters {parameter_difference:.2e}")
        record_property(f"parameters_{direction}.max_abs_diff", parameter_difference)
        for a, b in zip(leaves_sharded, leaves_single):
            np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-4, atol=1e-5)
        layers_sharded, layers_single = sparse_layers(sharded.nets[direction]), sparse_layers(single.nets[direction])
        assert layers_sharded and len(layers_sharded) == len(layers_single)
        for layer_sharded, layer_single in zip(layers_sharded, layers_single):
            array = support(layer_sharded)
            assert np.array_equal(np.asarray(array), np.asarray(support(layer_single)))
            assert array.sharding.is_fully_replicated and len(array.addressable_shards) == jax.device_count()
            first = np.asarray(array.addressable_shards[0].data)
            assert all(np.array_equal(np.asarray(shard.data), first) for shard in array.addressable_shards)

    checkpoints = tmp_path / "all" / "checkpoints"
    host_copy = tmp_path / "host_net_b_2_12.eqx"
    eqx.tree_serialise_leaves(host_copy, jax.device_get(sharded.nets["b"]))
    assert filecmp.cmp(checkpoints / "net_b_2_12.eqx", host_copy, shallow=False)

    ipf_config, model_config = configs(family, sparse_linear, devices=[0])
    resumed_config = dataclasses.replace(
        ipf_config, checkpoint_run=True, checkpoint_it=2, checkpoint_pass="f",
        checkpoint_f=str(checkpoints / "net_f_1_12.eqx"), sample_checkpoint_f=str(checkpoints / "sample_net_f_1_12.eqx"),
        checkpoint_b=str(checkpoints / "net_b_2_12.eqx"), sample_checkpoint_b=str(checkpoints / "sample_net_b_2_12.eqx"))
    resumed = IPF(resumed_config, model_config, IMAGES, str(tmp_path / "resumed"))
    # the b checkpoints hold the final state of the sharded run; the f checkpoints are those of iteration 1
    for loaded, original in zip(jax.tree.leaves(eqx.filter(resumed.nets["b"], eqx.is_array)),
                                jax.tree.leaves(eqx.filter(sharded.nets["b"], eqx.is_array))):
        assert loaded.sharding.is_fully_replicated and len(loaded.addressable_shards) == 1
        assert np.array_equal(np.asarray(loaded), np.asarray(original))
    for loaded, original in zip(jax.tree.leaves(resumed.ema["b"]), jax.tree.leaves(sharded.ema["b"])):
        assert np.array_equal(np.asarray(loaded), np.asarray(original))

    xs = jnp.asarray(np.random.default_rng(1).standard_normal((8, 3, 8, 8)), dtype=jnp.float32)
    ts = jnp.full((8,), 0.3, jnp.float32)
    np.testing.assert_allclose(np.asarray(evaluate(resumed.ema_net("b"), xs, ts)),
                               np.asarray(evaluate(sharded.ema_net("b"), xs, ts)), rtol=1e-5, atol=1e-6)
    resumed.plot_samples(12, 2, "b")
    a, b = grid(tmp_path / "resumed" / "im" / "b_2_12" / "im_grid_final.png"), grid(tmp_path / "all" / "im" / "b_2_12" / "im_grid_final.png")
    assert np.abs(a - b).max() <= 1 and float(np.mean(a != b)) < 0.01

    resumed.train()
    resumed_metrics = read_csv(tmp_path / "resumed" / "logs" / "metrics.csv")
    assert resumed_metrics.size == 4 and np.all(resumed_metrics["fb"] == "f") and np.all(resumed_metrics["n"] == 2)
    assert np.all(np.isfinite(resumed_metrics["loss"]))


def test_divisibility_is_checked(tmp_path):
    ipf_config, model_config = configs("st", "mask", devices=0)
    with pytest.raises(ValueError, match="cache_npar"):
        IPF(dataclasses.replace(ipf_config, cache_npar=jax.device_count() + 1), model_config, IMAGES, str(tmp_path))
    with pytest.raises(ValueError, match="devices"):
        IPF(dataclasses.replace(ipf_config, devices=[jax.device_count()]), model_config, IMAGES, str(tmp_path))
