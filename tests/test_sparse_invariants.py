"""Invariants of the sparse models through IPF training.

The fixed sparsity of a model is its support: the boolean mask of every st
SparseLinear (a dense weight under a mask) and the index array of every z1t
SparseLinear (K weights per output unit and the inputs they read). For both
families and both implementations the tests check that

  (i) a half-step of IPF leaves every support and the active parameter count
      unchanged and the EMA network carries the same support;
  (j) masked-out st weights receive an exactly zero gradient and never move;
  (k) every output unit of a sparse layer reads only its K inputs, and the
      masked and the gather layer compute the same map on the same support;
  (l) a checkpoint restores the support into a model built from another random
      key, through eqx.tree_deserialise_leaves and through IPF's resume path;
  (m) the residual output and context-prefix conventions hold after training.

The IPF runs train on the GPU:

    sbatch --job-name=validation slurm/run.sbatch python -m pytest -x -q tests/test_sparse_invariants.py
"""
from __future__ import annotations

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sparse_dsb.cache import minibatch
from sparse_dsb.ipf import IPF, IPFConfig
from sparse_dsb.models import ModelConfig, build_model, parameter_counts
from st.transformer import SparseLinear as MaskedSparseLinear
from z1t.components import SparseLinear as GatherSparseLinear

SPARSE_TYPES = (MaskedSparseLinear, GatherSparseLinear)
FAN_IN = 8
CASES = [("st", "mask"), ("st", "gather"), ("z1t", "mask"), ("z1t", "gather")]
MASKED_CASES = [case for case in CASES if case != ("z1t", "gather")]      # st blocks always use the masked layer
KEY = jax.random.PRNGKey(0)
gpu_only = pytest.mark.skipif(jax.default_backend() != "gpu", reason="training runs on the GPU only")


def model_config(family, sparse_linear, **overrides):
    return ModelConfig(family=family, image_size=8, patch_size=4, n_layers=2, n_embed=32, n_head=4, linear_fan_in=FAN_IN,
                       time_embed_dim=16, sparse_linear=sparse_linear, **overrides)


def ipf_config(**overrides):
    return IPFConfig(num_steps=6, n_ipf=1, num_iter=12, batch_size=16, cache_npar=8, num_cache_batches=2,
                     cache_refresh_stride=6, log_stride=3, gif_stride=6, plot_npar=4, lr=1e-3, **overrides, devices=1)


def synthetic_images(count=64):
    return np.asarray(jax.random.randint(KEY, (count, 8, 8, 3), 0, 256), dtype=np.uint8)


def sparse_layers(tree):
    """[(path, layer)] for every sparse linear of a model (or of a gradient tree of the same structure)."""
    flat, _ = jax.tree_util.tree_flatten_with_path(tree, is_leaf=lambda node: isinstance(node, SPARSE_TYPES))
    return [(jax.tree_util.keystr(path), node) for path, node in flat if isinstance(node, SPARSE_TYPES)]


def masked_layers(tree):
    return [(path, layer) for path, layer in sparse_layers(tree) if isinstance(layer, MaskedSparseLinear)]


def support_of(layer):
    """Boolean (n_out, n_in) matrix of the inputs each output unit reads."""
    if isinstance(layer, MaskedSparseLinear):
        return np.asarray(layer.mask)
    support = np.zeros((layer.out_features, layer.in_features), dtype=bool)
    np.put_along_axis(support, np.asarray(layer.indices), True, axis=1)
    return support


def supports(model):
    return [(path, support_of(layer)) for path, layer in sparse_layers(model)]


def assert_same_supports(expected, actual):
    assert [path for path, _ in expected] == [path for path, _ in actual]
    for (path, before), (_, after) in zip(expected, actual):
        assert np.array_equal(before, after), f"support changed in {path}"


def float_leaves(model):
    return [np.asarray(leaf) for leaf in jax.tree.leaves(eqx.filter(model, eqx.is_inexact_array))]


def count_changed(before, after):
    return sum(int(not np.array_equal(x, y)) for x, y in zip(before, after))


# ---------------------------------------------------------------- (i)
@gpu_only
@pytest.mark.parametrize("family, sparse_linear", CASES)
def test_half_step_keeps_support_and_active_count(tmp_path, record_property, family, sparse_linear):
    ipf = IPF(ipf_config(), model_config(family, sparse_linear), synthetic_images(), str(tmp_path))
    backward_before, forward_before = supports(ipf.nets["b"]), supports(ipf.nets["f"])
    counts = parameter_counts(ipf.nets["b"])
    floats_before = float_leaves(ipf.nets["b"])
    assert backward_before and all(np.all(support.sum(axis=1) == FAN_IN) for _, support in backward_before)

    ipf.ipf_step("b", 1)

    assert_same_supports(backward_before, supports(ipf.nets["b"]))
    assert_same_supports(backward_before, supports(ipf.ema_net("b")))
    assert_same_supports(forward_before, supports(ipf.nets["f"]))
    assert parameter_counts(ipf.nets["b"]) == counts
    changed = count_changed(floats_before, float_leaves(ipf.nets["b"]))
    assert changed == len(floats_before), f"only {changed} of {len(floats_before)} float leaves changed"
    record_property("sparse_layers", len(backward_before))
    record_property("parameters_stored", counts[0])
    record_property("parameters_active", counts[1])
    record_property("float_leaves_changed", f"{changed}/{len(floats_before)}")


# ---------------------------------------------------------------- (j)
@gpu_only
@pytest.mark.parametrize("family, sparse_linear", MASKED_CASES)
def test_masked_out_weights_have_zero_gradient_and_never_move(tmp_path, record_property, family, sparse_linear):
    ipf = IPF(ipf_config(), model_config(family, sparse_linear), synthetic_images(), str(tmp_path))
    masked = masked_layers(ipf.nets["b"])
    assert masked
    initial = {path: np.asarray(layer.linear.weight) for path, layer in masked}

    ipf.ipf_step("b", 1)

    trained = ipf.nets["b"]
    for path, layer in masked_layers(trained):
        mask, weight = np.asarray(layer.mask), np.asarray(layer.linear.weight)
        assert np.array_equal(weight[~mask], initial[path][~mask]), f"{path}: masked-out weights moved"
        assert not np.array_equal(weight[mask], initial[path][mask]), f"{path}: the support did not train"

    # The EMA update (1 - rate) * p + rate * e of an untouched weight (p = e = w) returns w only up to float32
    # rounding, so the EMA copy of a masked-out weight drifts by about one ulp per step; the reference EMAHelper
    # has the same formula and these weights are never read.
    ema_drift = 0.0
    for path, layer in masked_layers(ipf.ema_net("b")):
        mask, weight = np.asarray(layer.mask), np.asarray(layer.linear.weight)
        deviation = np.abs(weight[~mask] - initial[path][~mask]) / np.maximum(np.abs(initial[path][~mask]), 1e-30)
        ema_drift = max(ema_drift, float(deviation.max()))
    steps_taken = ipf.config.num_iter + 1
    assert ema_drift <= steps_taken * 4 * np.finfo(np.float32).eps, f"EMA masked-out weights drifted by {ema_drift:.2e}"
    record_property("ema_masked_out_relative_drift", ema_drift)

    cache = ipf.new_cache("b", 1)
    xs, outs, ts = minibatch(cache, jnp.arange(cache.xs.shape[0]), 0, ipf.config.batch_size)
    params, static = eqx.partition(trained, eqx.is_inexact_array)

    def mean_matching_loss(p):
        pred = jax.vmap(eqx.combine(p, static))(xs, ipf.T - ts) - xs
        return jnp.mean(jnp.square(pred - outs))

    grads = jax.grad(mean_matching_loss)(params)
    largest_support_gradient = 0.0
    for (path, layer), (_, grad_layer) in zip(masked, masked_layers(grads)):
        mask, gradient = np.asarray(layer.mask), np.asarray(grad_layer.linear.weight)
        assert np.all(gradient[~mask] == 0.0), f"{path}: nonzero gradient outside the support"
        assert np.any(gradient[mask] != 0.0), f"{path}: zero gradient on the support"
        largest_support_gradient = max(largest_support_gradient, float(np.abs(gradient[mask]).max()))
    record_property("masked_layers", len(masked))
    record_property("largest_support_gradient", largest_support_gradient)


# ---------------------------------------------------------------- (k)
@pytest.mark.parametrize("family, sparse_linear", CASES)
def test_sparse_units_read_only_their_support(record_property, family, sparse_linear):
    model = build_model(model_config(family, sparse_linear), KEY)
    layers = sparse_layers(model)
    assert layers
    implementations = {type(layer).__name__ for _, layer in layers}
    checked = 0
    for index, (path, layer) in enumerate(layers):
        support = support_of(layer)
        n_out, n_in = support.shape
        assert np.all(support.sum(axis=1) == FAN_IN), path
        weights = layer.linear.weight if isinstance(layer, MaskedSparseLinear) else layer.weight
        trainable = bool(jnp.any(weights != 0))                         # the output projection is zero-initialised
        x = jax.random.normal(jax.random.fold_in(KEY, index), (n_in,))
        y = np.asarray(layer(x))
        for j in sorted({0, n_in // 3, n_in // 2, n_in - 1}):
            changed = np.asarray(layer(x.at[j].add(1.0))) != y
            assert not np.any(changed & ~support[:, j]), f"{path}: input {j} reached a unit outside its support"
            if trainable:
                assert np.array_equal(changed, support[:, j]), f"{path}: input {j} did not reach every unit that reads it"
        checked += 1
    record_property("layers_checked", checked)
    record_property("implementations", ",".join(sorted(implementations)))


@pytest.mark.parametrize("family", ["st", "z1t"])
def test_mask_and_gather_layers_compute_the_same_map(check_close, family):
    model = build_model(model_config(family, "mask"), KEY)
    layers = masked_layers(model)
    assert layers
    for index, (path, layer) in enumerate(layers):
        mask = np.asarray(layer.mask)
        weight, bias = np.asarray(layer.linear.weight), np.asarray(layer.linear.bias)
        n_out, n_in = mask.shape
        indices = np.stack([np.flatnonzero(row) for row in mask])
        gather = GatherSparseLinear(n_in, n_out, k=FAN_IN, key=KEY)
        gather = eqx.tree_at(lambda l: (l.weight, l.bias, l.indices), gather,
                             (jnp.asarray(np.take_along_axis(weight, indices, axis=1)), jnp.asarray(bias), jnp.asarray(indices)))
        x = jax.random.normal(jax.random.fold_in(KEY, index), (n_in,))
        check_close(f"same_map[{family}]{path}", gather(x), layer(x))


# ---------------------------------------------------------------- (l)
@gpu_only
@pytest.mark.parametrize("family, sparse_linear", CASES)
def test_checkpoint_round_trip_restores_support(tmp_path, check_close, record_property, family, sparse_linear):
    config = model_config(family, sparse_linear)
    ipf = IPF(ipf_config(), config, synthetic_images(), str(tmp_path))
    ipf.ipf_step("b", 1)
    net_path = tmp_path / "checkpoints" / "net_b_1_12.eqx"
    ema_path = tmp_path / "checkpoints" / "sample_net_b_1_12.eqx"
    assert net_path.exists() and ema_path.exists()
    trained, trained_ema = ipf.nets["b"], ipf.ema_net("b")
    x, t = jax.random.uniform(KEY, (3, 8, 8)), jnp.float32(0.3)
    expected, expected_ema = trained(x, t), trained_ema(x, t)

    fresh = build_model(config, jax.random.PRNGKey(123))
    differing = sum(int(not np.array_equal(a, b)) for (_, a), (_, b) in zip(supports(fresh), supports(trained)))
    assert differing > 0, "a different key should give a different support"
    record_property("supports_differing_before_load", f"{differing}/{len(supports(fresh))}")

    loaded = eqx.tree_deserialise_leaves(str(net_path), fresh)
    loaded_ema = eqx.tree_deserialise_leaves(str(ema_path), fresh)
    assert_same_supports(supports(trained), supports(loaded))
    assert_same_supports(supports(trained_ema), supports(loaded_ema))
    check_close("round_trip.net_output", loaded(x, t), expected, rtol=0.0, atol=0.0)
    check_close("round_trip.ema_output", loaded_ema(x, t), expected_ema, rtol=0.0, atol=0.0)

    floats_only = eqx.combine(eqx.filter(loaded, eqx.is_inexact_array),
                              eqx.filter(fresh, lambda leaf: not eqx.is_inexact_array(leaf)))
    unpersisted_support = float(jnp.abs(floats_only(x, t) - expected).max())
    record_property("output_change_with_unpersisted_support", unpersisted_support)
    assert unpersisted_support > 1e-6, "restoring the floats but not the support should change the outputs"

    resumed = IPF(ipf_config(checkpoint_run=True, checkpoint_b=str(net_path), sample_checkpoint_b=str(ema_path), seed=1),
                  config, synthetic_images(), str(tmp_path / "resume"))
    assert_same_supports(supports(trained), supports(resumed.nets["b"]))
    assert_same_supports(supports(trained_ema), supports(resumed.ema_net("b")))
    check_close("resume.net_output", resumed.nets["b"](x, t), expected, rtol=0.0, atol=0.0)
    check_close("resume.ema_output", resumed.ema_net("b")(x, t), expected_ema, rtol=0.0, atol=0.0)


# ---------------------------------------------------------------- (m)
def without_residual_output(model, config):
    """The same parameters in a model that returns f(x, t) instead of x + f(x, t)."""
    template = build_model(dataclasses.replace(config, residual_output=False), KEY)
    return jax.tree_util.tree_unflatten(jax.tree_util.tree_structure(template), jax.tree_util.tree_leaves(model))


@gpu_only
@pytest.mark.parametrize("family, sparse_linear", CASES)
def test_residual_and_context_conventions_after_training(tmp_path, record_property, family, sparse_linear):
    config = model_config(family, sparse_linear)
    ipf = IPF(ipf_config(), config, synthetic_images(), str(tmp_path))
    x, t = jax.random.uniform(KEY, (3, 8, 8)), jnp.float32(0.3)
    assert np.array_equal(np.asarray(ipf.nets["b"](x, t)), np.asarray(x)), "the network is the identity at initialisation"

    ipf.ipf_step("b", 1)

    model = ipf.nets["b"]
    f = without_residual_output(model, config)(x, t)
    assert np.array_equal(np.asarray(model(x, t)), np.asarray(x + f))
    size = float(jnp.abs(f).max())
    record_property("f_max_abs_after_12_steps", size)
    assert 0.0 < size < float(jnp.abs(x).max())

    perturbed = x.at[:, 4:, 4:].add(1.0)
    shift = float(jnp.abs(model(perturbed, t)[:, :4, :4] - model(x, t)[:, :4, :4]).max())
    record_property("first_patch_shift_from_last_patch", shift)
    assert shift > 1e-6, "with the context prefix the first output patch sees the last input patch"
