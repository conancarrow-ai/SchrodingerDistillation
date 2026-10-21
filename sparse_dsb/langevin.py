"""Trajectory recording, ported from bridge/langevin.py of the dsb reference code.

Both routines run the chain for all num_steps steps with a lax.scan and record,
for every step k, the state after the step and the regression target

    out[k] = t_old - t_new,

where t_old is the mean used to produce the new state and t_new is the mean the
same map assigns to the new state (Proposition 3 of De Bortoli et al., 2021).
The recorded time stamp for step k is time[k] = sum_{j <= k} gammas[j].

Arrays are returned with the step axis first: x_tot and out have shape
(num_steps, batch, *shape) and steps has shape (num_steps,).
"""
from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp


def gaussian_gradient(x, mean_final, var_final):
    return -(x - mean_final) / var_final


def record_init_langevin(key, init_samples, gammas, time, mean_final, var_final):
    """Reference Ornstein-Uhlenbeck chain started at data samples (first backward pass)."""

    def step(x, inputs):
        gamma, step_key = inputs
        t_old = x + gamma * gaussian_gradient(x, mean_final, var_final)
        z = jax.random.normal(step_key, x.shape, x.dtype)
        x_new = t_old + jnp.sqrt(2.0 * gamma) * z
        t_new = x_new + gamma * gaussian_gradient(x_new, mean_final, var_final)
        return x_new, (x_new, t_old - t_new)

    keys = jax.random.split(key, gammas.shape[0])
    _, (x_tot, out) = jax.lax.scan(step, init_samples, (gammas, keys))
    return x_tot, out, time


def record_langevin_seq(key, batched_net: Callable, init_samples, gammas, time, mean_match: bool = True, sample: bool = False):
    """Chain driven by a learnt network: batched_net(x, t) maps (B, *shape), (B,) -> (B, *shape).

    With mean_match the network output is the mean of the next state; otherwise the
    mean is x + network output. With sample=True the last step adds no noise.
    """
    num_steps = gammas.shape[0]

    def mean_of(x, t):
        batch_t = jnp.full((x.shape[0],), t, x.dtype)
        output = batched_net(x, batch_t)
        return output if mean_match else x + output

    def step(x, inputs):
        gamma, t, step_key, is_last = inputs
        t_old = mean_of(x, t)
        z = jax.random.normal(step_key, x.shape, x.dtype)
        noise_scale = jnp.where(sample & is_last, 0.0, jnp.sqrt(2.0 * gamma))
        x_new = t_old + noise_scale * z
        t_new = mean_of(x_new, t)
        return x_new, (x_new, t_old - t_new)

    keys = jax.random.split(key, num_steps)
    is_last = jnp.arange(num_steps) == num_steps - 1
    _, (x_tot, out) = jax.lax.scan(step, init_samples, (gammas, time, keys, is_last))
    return x_tot, out, time
