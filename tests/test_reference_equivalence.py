"""Equivalence of the JAX port with the PyTorch reference code (bridge/ of the dsb submodule).

Every test feeds identical inputs to both implementations and compares the
results with float32 tolerances (rtol = atol = 1e-5 unless stated). The
reference runs with CPU torch (the `validation` extra of pyproject.toml). Only
its torch-only modules are imported: bridge.langevin, bridge.data.cacheloader,
bridge.runners.ema and bridge.models.unet.layers; the package __init__ of
bridge.data imports torchvision, so an empty namespace package stands in for it.

Noise replay: the port splits its key into num_steps keys and draws
jax.random.normal(step_key, x.shape, x.dtype) at step k. The tests draw those
arrays in advance and patch torch.randn to return them in order, so both
chains see the same Gaussian variates and can be compared row by row.

    sbatch --job-name=validation slurm/run.sbatch python -m pytest -x -q tests/test_reference_equivalence.py
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
import optax
import pytest

torch = pytest.importorskip("torch")

from sparse_dsb.cache import Cache, concatenate_caches, flatten_trajectories
from sparse_dsb.ipf import IPF, IPFConfig
from sparse_dsb.langevin import record_init_langevin, record_langevin_seq
from sparse_dsb.models import ModelConfig, sinusoidal_embedding
from sparse_dsb.schedule import cumulative_time, make_gammas

REFERENCE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "diffusion_schrodinger_bridge")
SHAPE = (3, 8, 8)
MEAN_FINAL, VAR_FINAL = 0.445, 0.5
BATCH = 4
KEY = jax.random.PRNGKey(0)


def load_reference():
    if REFERENCE not in sys.path:
        sys.path.insert(0, REFERENCE)
    if "bridge.data" not in sys.modules:
        importlib.import_module("bridge")
        package = types.ModuleType("bridge.data")
        package.__path__ = [os.path.join(REFERENCE, "bridge", "data")]
        sys.modules["bridge.data"] = package
    return types.SimpleNamespace(
        langevin=importlib.import_module("bridge.langevin"),
        cacheloader=importlib.import_module("bridge.data.cacheloader"),
        ema=importlib.import_module("bridge.runners.ema"),
        layers=importlib.import_module("bridge.models.unet.layers"),
    )


reference = load_reference()


# ---------------------------------------------------------------- reference and port constructions
def reference_schedule(num_steps, gamma_min=1e-5, gamma_max=0.1, gamma_space="linspace"):
    """gammas and T as IPFBase.__init__ builds them: a numpy ramp mirrored, a float64 tensor, T = sum."""
    half = num_steps // 2
    if gamma_space == "linspace":
        ramp = np.linspace(gamma_min, gamma_max, half)
    else:
        ramp = np.geomspace(gamma_min, gamma_max, half)
    gammas = torch.tensor(np.concatenate([ramp, np.flip(ramp)]))
    return gammas, torch.sum(gammas)


def reference_langevin(num_steps, gammas, mean_match=True):
    """The reference Langevin with the adaptive prior mean broadcast to the image shape, as get_datasets builds it."""
    return reference.langevin.Langevin(num_steps, SHAPE, gammas, None, device="cpu",
                                       mean_final=torch.full(SHAPE, MEAN_FINAL), var_final=torch.tensor(VAR_FINAL),
                                       mean_match=mean_match)


def port_schedule(num_steps, gamma_min=1e-5, gamma_max=0.1, gamma_space="linspace"):
    gammas = make_gammas(num_steps, gamma_min, gamma_max, gamma_space)
    return jnp.asarray(gammas), jnp.asarray(cumulative_time(gammas)), float(gammas.sum())


def port_noise(key, num_steps, shape):
    """The arrays the port draws at steps 0 .. num_steps - 1 from key."""
    return [np.asarray(jax.random.normal(step_key, shape, jnp.float32)) for step_key in jax.random.split(key, num_steps)]


class NoiseReplay:
    """Stands in for torch.randn and returns the given arrays in order."""

    def __init__(self, draws):
        self.draws = [torch.from_numpy(np.ascontiguousarray(draw)) for draw in draws]
        self.calls = 0

    def __call__(self, *size, **kwargs):
        if len(size) == 1 and not isinstance(size[0], int):
            size = tuple(size[0])
        draw = self.draws[self.calls]
        self.calls += 1
        assert tuple(draw.shape) == tuple(size), (draw.shape, size)
        return draw.clone()


class ToyTorchNet(torch.nn.Module):
    """y = a * x + b * t + c with t given as a (B, 1) tensor, as the reference calls its networks."""

    def __init__(self, a, b, c):
        super().__init__()
        self.a = torch.nn.Parameter(torch.from_numpy(a.copy()))
        self.b = torch.nn.Parameter(torch.from_numpy(b.copy()))
        self.c = torch.nn.Parameter(torch.from_numpy(c.copy()))

    def forward(self, x, t):
        t = t.reshape(x.shape[0], 1, 1, 1)
        return self.a * x + self.b * t + self.c


class ToyJaxNet(eqx.Module):
    """The same map for one sample; the port batches networks with jax.vmap."""
    a: jax.Array
    b: jax.Array
    c: jax.Array

    def __call__(self, x, t):
        return self.a * x + self.b * t + self.c


def toy_coefficients(mean_match: bool, key=jax.random.PRNGKey(7)):
    """a near 0.9 (the network is the next mean) or near -0.1 (the network is a residual), b and c small."""
    keys = jax.random.split(key, 3)
    centre = 0.9 if mean_match else -0.1
    a = centre + 0.05 * jax.random.normal(keys[0], SHAPE)
    b = 0.1 * jax.random.normal(keys[1], SHAPE)
    c = 0.05 * jax.random.normal(keys[2], SHAPE)
    return tuple(np.asarray(value, dtype=np.float32) for value in (a, b, c))


def batched_toy_net(a, b, c):
    net = ToyJaxNet(jnp.asarray(a), jnp.asarray(b), jnp.asarray(c))
    return lambda xs, ts: jax.vmap(net)(xs, ts)


def steps_to_rows(time, batch):
    """The (batch, num_steps, 1) time stamps the reference returns, from the port's (num_steps,) vector."""
    return np.broadcast_to(np.asarray(time)[None, :, None], (batch, time.shape[0], 1))


# ---------------------------------------------------------------- (a) schedule
@pytest.mark.parametrize("num_steps, gamma_space", [(50, "linspace"), (20, "linspace"), (50, "geomspace")])
def test_schedule_matches_reference(check_close, num_steps, gamma_space):
    gammas_reference, T_reference = reference_schedule(num_steps, gamma_space=gamma_space)
    langevin = reference_langevin(num_steps, gammas_reference)
    gammas, time, T = port_schedule(num_steps, gamma_space=gamma_space)
    tag = f"schedule[{num_steps},{gamma_space}]"
    check_close(f"{tag}.gammas", gammas, langevin.gammas.numpy())
    check_close(f"{tag}.time", time, langevin.time.numpy())
    check_close(f"{tag}.T", np.float64(T), float(T_reference))
    assert float(time[-1]) == pytest.approx(T, rel=1e-6)


# ---------------------------------------------------------------- (b) reference chain from data
def test_record_init_langevin_matches_reference(check_close, monkeypatch):
    num_steps = 50
    gammas_reference, _ = reference_schedule(num_steps)
    langevin = reference_langevin(num_steps, gammas_reference)
    x0 = np.asarray(jax.random.uniform(KEY, (BATCH,) + SHAPE), dtype=np.float32)
    noise_key = jax.random.PRNGKey(11)
    replay = NoiseReplay(port_noise(noise_key, num_steps, (BATCH,) + SHAPE))
    monkeypatch.setattr(torch, "randn", replay)
    x_reference, out_reference, steps_reference = langevin.record_init_langevin(torch.from_numpy(x0))
    assert replay.calls == num_steps

    gammas, time, _ = port_schedule(num_steps)
    x_tot, out, steps = record_init_langevin(noise_key, jnp.asarray(x0), gammas, time, jnp.float32(MEAN_FINAL), jnp.float32(VAR_FINAL))
    assert tuple(x_reference.shape) == (BATCH, num_steps) + SHAPE and x_tot.shape == (num_steps, BATCH) + SHAPE
    check_close("init_langevin.x_tot", x_tot, np.swapaxes(x_reference.numpy(), 0, 1))
    check_close("init_langevin.out", out, np.swapaxes(out_reference.numpy(), 0, 1))
    assert tuple(steps_reference.shape) == (BATCH, num_steps, 1)
    check_close("init_langevin.steps", steps_to_rows(steps, BATCH), steps_reference.numpy())


# ---------------------------------------------------------------- (c) chain driven by a network
@pytest.mark.parametrize("sample", [False, True])
@pytest.mark.parametrize("mean_match", [True, False])
def test_record_langevin_seq_matches_reference(check_close, monkeypatch, mean_match, sample):
    num_steps = 50
    gammas_reference, _ = reference_schedule(num_steps)
    langevin = reference_langevin(num_steps, gammas_reference, mean_match=mean_match)
    a, b, c = toy_coefficients(mean_match)
    x0 = np.asarray(jax.random.normal(KEY, (BATCH,) + SHAPE), dtype=np.float32)
    noise_key = jax.random.PRNGKey(12)
    replay = NoiseReplay(port_noise(noise_key, num_steps, (BATCH,) + SHAPE))
    monkeypatch.setattr(torch, "randn", replay)
    with torch.no_grad():
        x_reference, out_reference, steps_reference = langevin.record_langevin_seq(ToyTorchNet(a, b, c), torch.from_numpy(x0), sample=sample)
    assert replay.calls == num_steps - (1 if sample else 0), "the reference adds no noise at the last step when sampling"

    gammas, time, _ = port_schedule(num_steps)
    x_tot, out, steps = record_langevin_seq(noise_key, batched_toy_net(a, b, c), jnp.asarray(x0), gammas, time,
                                            mean_match=mean_match, sample=sample)
    tag = f"langevin_seq[mean_match={mean_match},sample={sample}]"
    check_close(f"{tag}.x_tot", x_tot, np.swapaxes(x_reference.numpy(), 0, 1))
    check_close(f"{tag}.out", out, np.swapaxes(out_reference.numpy(), 0, 1))
    check_close(f"{tag}.steps", steps_to_rows(steps, BATCH), steps_reference.numpy())


# ---------------------------------------------------------------- (d) cache flattening
class OneDirectionLoader:
    """The repeater interface of the reference data loaders: next() yields a (batch,) tuple."""

    def __init__(self, batches):
        self.batches = iter(batches)

    def __next__(self):
        return (next(self.batches),)


@pytest.mark.parametrize("fb, n, num_batches", [("b", 1, 1), ("b", 2, 2), ("f", 2, 1)])
def test_cacheloader_flattening_matches_reference(check_close, monkeypatch, fb, n, num_batches):
    num_steps, batch = 20, 3
    gammas_reference, _ = reference_schedule(num_steps)
    langevin = reference_langevin(num_steps, gammas_reference)
    a, b, c = toy_coefficients(True)
    gammas, time, _ = port_schedule(num_steps)
    net = batched_toy_net(a, b, c)
    mean_final, var_final = jnp.float32(MEAN_FINAL), jnp.float32(VAR_FINAL)

    batches = [np.asarray(jax.random.uniform(k, (batch,) + SHAPE), dtype=np.float32) for k in jax.random.split(KEY, num_batches)]
    draws, parts = [], []
    for index, chain_key in enumerate(jax.random.split(jax.random.PRNGKey(13), num_batches)):
        if fb == "f":
            prior_key, chain_key = jax.random.split(chain_key)
            z0 = jax.random.normal(prior_key, (batch,) + SHAPE)
            draws.append(np.asarray(z0))
            x_init = mean_final + jnp.sqrt(var_final) * z0      # IPF.prior_batch
        else:
            x_init = jnp.asarray(batches[index])
        draws.extend(port_noise(chain_key, num_steps, (batch,) + SHAPE))
        if fb == "b" and n == 1:
            parts.append(flatten_trajectories(*record_init_langevin(chain_key, x_init, gammas, time, mean_final, var_final)))
        else:
            parts.append(flatten_trajectories(*record_langevin_seq(chain_key, net, x_init, gammas, time, mean_match=True)))
    cache = concatenate_caches(parts) if num_batches > 1 else parts[0]

    replay = NoiseReplay(draws)
    monkeypatch.setattr(torch, "randn", replay)
    loader = reference.cacheloader.CacheLoader(
        fb, ToyTorchNet(a, b, c), OneDirectionLoader(torch.from_numpy(x) for x in batches), num_batches, langevin, n,
        mean=torch.full(SHAPE, MEAN_FINAL), std=torch.sqrt(torch.tensor(VAR_FINAL)), batch_size=batch, device="cpu")
    assert replay.calls == len(draws)
    rows = num_batches * batch * num_steps
    assert tuple(loader.data.shape) == (rows, 2) + SHAPE and len(loader) == rows == cache.xs.shape[0]
    tag = f"cacheloader[{fb},n={n},batches={num_batches}]"
    check_close(f"{tag}.x", cache.xs, loader.data[:, 0].numpy())
    check_close(f"{tag}.out", cache.outs, loader.data[:, 1].numpy())
    check_close(f"{tag}.steps", cache.steps, loader.steps_data[:, 0].numpy())
    x_row, out_row, steps_row = loader[num_steps + 2]
    check_close(f"{tag}.getitem", np.concatenate([np.asarray(cache.xs[num_steps + 2]).ravel(), np.asarray(cache.outs[num_steps + 2]).ravel(),
                                                 np.asarray(cache.steps[num_steps + 2]).ravel()]),
                np.concatenate([x_row.numpy().ravel(), out_row.numpy().ravel(), steps_row.numpy().ravel()]))


# ---------------------------------------------------------------- (e) (f) (g) one training step of the port
def tiny_ipf(tmp_path, **overrides):
    fields = dict(num_steps=50, batch_size=BATCH, lr=1e-4, grad_clip=1.0, ema_rate=0.999, log_stride=1)
    fields.update(overrides)
    config = IPFConfig(**fields)
    model_config = ModelConfig(image_size=8, patch_size=4, n_layers=1, n_embed=16, n_head=2, linear_fan_in=8, time_embed_dim=8)
    return IPF(config, model_config, np.zeros((4, 8, 8, 3), dtype=np.uint8), str(tmp_path))


@pytest.mark.parametrize("grad_clip, lr", [(1.0, 1e-4), (0.02, 1e-4), (1.0, 0.1)])
@pytest.mark.parametrize("mean_match", [True, False])
def test_train_step_matches_reference(tmp_path, check_close, record_property, mean_match, grad_clip, lr):
    """IPF._train_step (loss, gradient norm, clipped Adam update and EMA update) against the reference
    ipf_step body on the same three minibatches, with the toy network in place of the transformer."""
    steps = 3
    rows = steps * BATCH
    ipf = tiny_ipf(tmp_path, mean_match=mean_match, grad_clip=grad_clip, lr=lr)
    _, T_reference = reference_schedule(ipf.config.num_steps)
    a, b, c = toy_coefficients(mean_match)
    keys = jax.random.split(jax.random.PRNGKey(21), 3)
    xs = np.asarray(jax.random.uniform(keys[0], (rows,) + SHAPE), dtype=np.float32)
    outs = np.asarray(0.1 * jax.random.normal(keys[1], (rows,) + SHAPE), dtype=np.float32)
    ts = np.asarray(jax.random.uniform(keys[2], (rows,), minval=0.0, maxval=ipf.T), dtype=np.float32)
    cache = Cache(jnp.asarray(xs), jnp.asarray(outs), jnp.asarray(ts))
    permutation = jnp.arange(rows)

    params, static = eqx.partition(ToyJaxNet(jnp.asarray(a), jnp.asarray(b), jnp.asarray(c)), eqx.is_inexact_array)
    opt_state = ipf.optimizer.init(params)
    ema_params = params

    net = ToyTorchNet(a, b, c)
    optimizer = torch.optim.Adam(net.parameters(), lr=ipf.config.lr)
    ema = reference.ema.EMAHelper(mu=ipf.config.ema_rate)
    ema.register(net)

    tag = f"train_step[mean_match={mean_match},clip={grad_clip}]"
    for step in range(steps):
        rows_of_step = slice(step * BATCH, (step + 1) * BATCH)
        x, out = torch.from_numpy(xs[rows_of_step]), torch.from_numpy(outs[rows_of_step])
        eval_steps = T_reference - torch.from_numpy(ts[rows_of_step])[:, None]
        pred = net(x, eval_steps) - x if mean_match else net(x, eval_steps)
        loss = torch.nn.functional.mse_loss(pred, out)
        optimizer.zero_grad()
        loss.backward()
        total_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
        optimizer.step()
        ema.update(net)

        params, opt_state, ema_params, loss_port, grad_norm = ipf._train_step(
            params, static, opt_state, ema_params, cache, permutation, jnp.int32(step))
        record_property(f"{tag}.clipped[{step}]", bool(total_norm.item() > grad_clip))
        check_close(f"{tag}.loss[{step}]", loss_port, loss.item())
        check_close(f"{tag}.grad_norm[{step}]", grad_norm, total_norm.item())
        for name in ("a", "b", "c"):
            check_close(f"{tag}.{name}[{step}]", getattr(params, name), getattr(net, name).detach().numpy())
            check_close(f"{tag}.ema_{name}[{step}]", getattr(ema_params, name), ema.shadow[name].numpy())


def test_ema_update_matches_reference(check_close):
    rate = 0.999
    a, b, c = toy_coefficients(True)
    net = ToyTorchNet(a, b, c)
    helper = reference.ema.EMAHelper(mu=rate)
    helper.register(net)
    ema_tree = {"a": jnp.asarray(a), "b": jnp.asarray(b), "c": jnp.asarray(c)}
    for step, key in enumerate(jax.random.split(jax.random.PRNGKey(5), 6)):
        moved = {name: np.asarray(jnp.asarray(value) + 0.1 * jax.random.normal(jax.random.fold_in(key, index), SHAPE), dtype=np.float32)
                 for index, (name, value) in enumerate(ema_tree.items())}
        with torch.no_grad():
            for name, value in moved.items():
                getattr(net, name).copy_(torch.from_numpy(value))
        helper.update(net)
        ema_tree = jax.tree.map(lambda e, p: (1.0 - rate) * p + rate * e, ema_tree,
                                {name: jnp.asarray(value) for name, value in moved.items()})      # IPF._compile.train_step
        for name in ema_tree:
            check_close(f"ema.{name}[{step}]", ema_tree[name], helper.shadow[name].numpy())


def test_adam_with_clipping_matches_reference(check_close, record_property):
    """torch.optim.Adam(lr=1e-4) after clip_grad_norm_(1.0) against optax.chain(clip_by_global_norm(1.0), adam(1e-4)),
    three consecutive steps with gradient norms 3 (clipped), 0.5 (not clipped) and 10 (clipped). Parameters are of
    size 1e-2 so that float32 rounding of a parameter stays far below the 1e-4 size of an update."""
    lr, clip = 1e-4, 1.0
    shapes = [(3, 8, 8), (16,)]
    keys = jax.random.split(jax.random.PRNGKey(9), 4)
    initial = [np.asarray(0.01 * jax.random.normal(jax.random.fold_in(keys[0], index), shape), dtype=np.float32)
               for index, shape in enumerate(shapes)]
    gradients = []
    for step, norm in enumerate([3.0, 0.5, 10.0]):
        raw = [np.asarray(jax.random.normal(jax.random.fold_in(keys[1 + step], index), shape), dtype=np.float64)
               for index, shape in enumerate(shapes)]
        scale = norm / np.sqrt(sum(float(np.sum(g * g)) for g in raw))
        gradients.append([np.asarray(g * scale, dtype=np.float32) for g in raw])

    tensors = [torch.nn.Parameter(torch.from_numpy(p.copy())) for p in initial]
    torch_optimizer = torch.optim.Adam(tensors, lr=lr)
    optimizer = optax.chain(optax.clip_by_global_norm(clip), optax.adam(lr))
    params = [jnp.asarray(p) for p in initial]
    state = optimizer.init(params)
    for step, grads in enumerate(gradients):
        torch_before = [tensor.detach().numpy().astype(np.float64) for tensor in tensors]
        for tensor, g in zip(tensors, grads):
            tensor.grad = torch.from_numpy(g.copy())
        total_norm = torch.nn.utils.clip_grad_norm_(tensors, clip)
        torch_optimizer.step()
        torch_after = [tensor.detach().numpy().astype(np.float64) for tensor in tensors]

        jax_grads = [jnp.asarray(g) for g in grads]
        updates, state = optimizer.update(jax_grads, state, params)
        new_params = eqx.apply_updates(params, updates)
        check_close(f"adam.grad_norm[{step}]", optax.global_norm(jax_grads), total_norm.item())
        record_property(f"adam.clipped[{step}]", bool(total_norm.item() > clip))
        for index, (new, old) in enumerate(zip(new_params, params)):
            check_close(f"adam.params[{step}][{index}]", new, torch_after[index], rtol=1e-5, atol=1e-8)
            port_update = np.asarray(new, np.float64) - np.asarray(old, np.float64)
            record_property(f"adam.update_scale[{step}][{index}]", float(np.abs(port_update).max()))
            check_close(f"adam.update[{step}][{index}]", port_update, torch_after[index] - torch_before[index], rtol=1e-5, atol=1e-8)
        params = new_params


# ---------------------------------------------------------------- (h) time embedding
@pytest.mark.parametrize("dim", [64, 33])
def test_time_embedding_matches_reference(check_close, dim):
    t = np.array([0.0, 1e-5, 0.013, 0.5, 1.0, 2.5, 2.49999, 10.0], dtype=np.float32)
    expected = reference.layers.timestep_embedding(torch.from_numpy(t), dim).numpy()
    port = jax.vmap(lambda s: sinusoidal_embedding(s, dim))(jnp.asarray(t))
    assert port.shape == (t.shape[0], dim)
    check_close(f"time_embedding[{dim}]", port, expected)
