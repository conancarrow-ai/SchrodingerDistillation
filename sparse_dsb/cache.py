"""The trajectory cache of bridge/data/cacheloader.py, kept on the device.

A cache holds M = num_cache_batches * cache_batch_size * num_steps rows. Row
b * num_steps + k is the state after step k of trajectory b together with its
regression target and time stamp, the flattening of the reference code. Rows are
visited in a random permutation drawn at each refresh instead of in storage
order (the reference code iterates the cache in storage order).
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp


class Cache(NamedTuple):
    xs: jax.Array      # (M, *shape)
    outs: jax.Array    # (M, *shape)
    steps: jax.Array   # (M,)


def flatten_trajectories(x_tot, out, time) -> Cache:
    """(num_steps, B, *shape) -> rows ordered by trajectory then step."""
    num_steps, batch = x_tot.shape[:2]
    xs = jnp.swapaxes(x_tot, 0, 1).reshape((batch * num_steps,) + x_tot.shape[2:])
    outs = jnp.swapaxes(out, 0, 1).reshape((batch * num_steps,) + out.shape[2:])
    steps = jnp.tile(time, batch)
    return Cache(xs, outs, steps)


def concatenate_caches(caches: list[Cache]) -> Cache:
    return Cache(*(jnp.concatenate(parts, axis=0) for parts in zip(*caches)))


def minibatch(cache: Cache, permutation, iteration, batch_size):
    """Rows of the cache at positions [iteration * batch_size, (iteration + 1) * batch_size) of the permutation (wrapping around)."""
    rows = permutation.shape[0]
    index = permutation[(iteration * batch_size + jnp.arange(batch_size)) % rows]
    return cache.xs[index], cache.outs[index], cache.steps[index]
