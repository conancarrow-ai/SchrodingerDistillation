"""Micro-benchmark of the z1t path: full model train step vs isolated components."""
import time, functools
import jax, jax.numpy as jnp, equinox as eqx
from sparse_dsb.models import ModelConfig, build_model
from z1t.components import _dwconv1d_causal

def timeit(name, fn, *args, repeats=5):
    out = fn(*args); jax.block_until_ready(out)
    t0 = time.perf_counter()
    for _ in range(repeats):
        out = fn(*args)
    jax.block_until_ready(out)
    print(f"{name:55s} {(time.perf_counter() - t0) / repeats * 1e3:9.1f} ms", flush=True)

key = jax.random.PRNGKey(0)
B, C, T = 128, 256, 513
x = jax.random.normal(key, (B, 3, 64, 64)); t = jnp.full((B,), 1.0)

for family, extra in (("st", {}), ("z1t", {"aft_kind": "conv", "n_head": 8}), ("z1t", {"aft_kind": "conv", "n_head": 8, "sparse_linear": "gather"}), ("z1t", {"aft_kind": "recurrence"})):
    cfg = ModelConfig(family=family, image_size=64, patch_size=4, n_layers=6, n_embed=256, linear_fan_in=64, **({"n_head": 4} | extra))
    model = build_model(cfg, key)
    params, static = eqx.partition(model, eqx.is_inexact_array)
    @jax.jit
    def fwd(p, x, t):
        return jax.vmap(eqx.combine(p, static))(x, t)
    @jax.jit
    def step(p, x, t):
        def loss(p):
            return jnp.mean(jnp.square(jax.vmap(eqx.combine(p, static))(x, t) - x))
        return jax.value_and_grad(loss)(p)
    label = f"{family} {extra.get('aft_kind', '')} {extra.get('sparse_linear', 'mask')}"
    timeit(f"{label} forward B={B}", fwd, params, x, t)
    timeit(f"{label} value_and_grad B={B}", step, params, x, t)

# isolated AFT-conv pieces at the model's shapes: per sample (C, T) with kernel (C, 4)
xs = jax.random.normal(key, (B, C, T)); kernel = jax.random.normal(key, (C, 4))
dw = jax.jit(jax.vmap(_dwconv1d_causal, in_axes=(0, None)))
timeit("depthwise causal conv forward (vmap B)", dw, xs, kernel)
dw_grad = jax.jit(jax.grad(lambda xs, k: jnp.sum(jax.vmap(_dwconv1d_causal, in_axes=(0, None))(xs, k)), argnums=(0, 1)))
timeit("depthwise causal conv grad (x and kernel)", dw_grad, xs, kernel)
cs = jax.jit(lambda a: jnp.cumsum(a, axis=1))
ys = jax.random.normal(key, (B, T, C))
timeit("cumsum over T forward", cs, ys)
cs_grad = jax.jit(jax.grad(lambda a: jnp.sum(jnp.cumsum(a, axis=1) ** 2)))
timeit("cumsum over T grad", cs_grad, ys)
# shift-and-add formulation of the same causal conv (for comparison only)
def shift_conv(x, kernel):
    C, T = x.shape; s = kernel.shape[-1]
    out = jnp.zeros_like(x)
    for j in range(s):
        shifted = jnp.pad(x, ((0, 0), (j, 0)))[:, :T]
        out = out + kernel[:, s - 1 - j][:, None] * shifted
    return out
sc = jax.jit(jax.vmap(shift_conv, in_axes=(0, None)))
timeit("shift-and-add causal conv forward", sc, xs, kernel)
sc_grad = jax.jit(jax.grad(lambda xs, k: jnp.sum(jax.vmap(shift_conv, in_axes=(0, None))(xs, k)), argnums=(0, 1)))
timeit("shift-and-add causal conv grad", sc_grad, xs, kernel)
ref = dw(xs, kernel); alt = sc(xs, kernel); print("max abs diff conv vs shift-add:", float(jnp.abs(ref - alt).max()))
