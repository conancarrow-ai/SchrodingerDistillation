"""Iterative proportional fitting with cached trajectories: a JAX port of
IPFSequential (bridge/runners/ipf.py) of the dsb reference code.

One IPF iteration n is a backward half-step followed by a forward half-step.
A half-step for direction d (b or f) trains the network of direction d on
trajectories simulated with the exponential moving average of the opposite
network (or, for n = 1 and d = b, with the reference Ornstein-Uhlenbeck chain):

    loss = mean | net_d(x, T - t) - x - out |^2        (mean matching)

over rows (x, out, t) of a trajectory cache that is rebuilt every
cache_refresh_stride gradient steps. Each half-step starts from the previous
weights of net_d (use_prev_net) or from a fresh initialisation, with a fresh
Adam state in both cases, and updates an exponential moving average that is
used for simulation, sampling and checkpoints (sample_net_*). Sampling calls
net_d(x, t) with the cumulative time t of the step, as the reference code does.

Semantics kept from the reference code: the symmetric gamma schedule and its
cumulative time, the regression targets, the T - t network input during
training and t during simulation, the EMA rate and its update rule, Adam with
gradient-norm clipping, fresh optimisers per half-step, the cache refresh
schedule, the prior N(mean_final, var_final), the adaptive prior mean, the
random horizontal flip and the checkpoint and sample-grid cadence. Differences:
cache rows are visited in a random permutation drawn at each refresh instead
of in storage order, the whole cache lives on the accelerator, and simulation
and training steps are jitted lax.scan / optax programs.
"""
from __future__ import annotations

import csv
import dataclasses
import functools
import json
import os
import time
from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from .cache import Cache, concatenate_caches, flatten_trajectories, minibatch
from .data import adaptive_mean, sample_images
from .langevin import record_init_langevin, record_langevin_seq
from .models import ModelConfig, build_model, parameter_counts
from .plots import save_image_grid
from .schedule import cumulative_time, make_gammas


@dataclass
class IPFConfig:
    # diffusion schedule
    num_steps: int = 50
    gamma_min: float = 1e-5
    gamma_max: float = 0.1
    gamma_space: str = "linspace"
    mean_match: bool = True
    # prior N(mean_final, var_final)
    var_final: float = 0.5
    adaptive_mean: bool = True
    mean_final: float = 0.0
    # optimisation
    n_ipf: int = 20
    num_iter: int = 50000
    batch_size: int = 128
    lr: float = 1e-4
    grad_clipping: bool = True
    grad_clip: float = 1.0
    ema: bool = True
    ema_rate: float = 0.999
    use_prev_net: bool = True
    # trajectory cache
    cache_npar: int = 300
    num_cache_batches: int = 1
    cache_refresh_stride: int = 100
    # data
    random_flip: bool = True
    # logging, checkpoints, sample grids
    log_stride: int = 10
    gif_stride: int = 5000
    plot_npar: int = 100
    # resume (checkpoint_pass is b or f)
    checkpoint_run: bool = False
    checkpoint_it: int = 1
    checkpoint_pass: str = "b"
    checkpoint_f: str = ""
    checkpoint_b: str = ""
    sample_checkpoint_f: str = ""
    sample_checkpoint_b: str = ""
    seed: int = 0


class CSVLog:
    def __init__(self, path: str, fields: list[str]):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.path, self.fields = path, fields
        self.handle = open(path, "a", newline="")
        self.writer = csv.DictWriter(self.handle, fieldnames=fields)
        if self.handle.tell() == 0:
            self.writer.writeheader()

    def log(self, **row):
        self.writer.writerow({k: row.get(k, "") for k in self.fields})
        self.handle.flush()


class IPF:
    def __init__(self, config: IPFConfig, model_config: ModelConfig, images_uint8, out_dir: str):
        self.config, self.model_config = config, model_config
        self.out_dir = out_dir
        for sub in ("im", "checkpoints", "logs"):
            os.makedirs(os.path.join(out_dir, sub), exist_ok=True)
        self.logger = CSVLog(os.path.join(out_dir, "logs", "metrics.csv"),
                             ["fb", "n", "i", "global_step", "loss", "grad_norm", "elapsed"])
        self.plot_logger = CSVLog(os.path.join(out_dir, "logs", "plot_logs.csv"),
                                  ["fb", "n", "i", "init_var", "final_var", "mean_init", "mean_final", "T"])

        gammas = make_gammas(config.num_steps, config.gamma_min, config.gamma_max, config.gamma_space)
        self.gammas = jnp.asarray(gammas)
        self.time = jnp.asarray(cumulative_time(gammas))
        self.T = float(gammas.sum())

        self.images = jnp.asarray(images_uint8)
        self.shape = (model_config.channels, model_config.image_size, model_config.image_size)
        mean = adaptive_mean(images_uint8) if config.adaptive_mean else config.mean_final
        self.mean_final = jnp.float32(mean)
        self.var_final = jnp.float32(config.var_final)
        self.std_final = jnp.sqrt(self.var_final)

        self.key = jax.random.PRNGKey(config.seed)
        self.nets, self.ema = {}, {}
        for direction in ("f", "b"):
            self.build_net(direction)
        if config.checkpoint_run:
            self.load_checkpoints()

        clip = optax.clip_by_global_norm(config.grad_clip) if config.grad_clipping else optax.identity()
        self.optimizer = optax.chain(clip, optax.adam(config.lr))
        self._compile()

        stored, active = parameter_counts(self.nets["b"])
        self.info = {"parameters_stored": stored, "parameters_active": active, "T": self.T,
                     "mean_final": float(mean), "var_final": config.var_final,
                     "sequence": model_config.sequence, "tokens": model_config.tokens}
        with open(os.path.join(out_dir, "info.json"), "w") as handle:
            json.dump(self.info, handle, indent=2)
        print(f"parameters stored={stored:,d} active={active:,d} sequence={model_config.sequence} T={self.T:.4f} "
              f"mean_final={float(mean):.4f}", flush=True)

    # ------------------------------------------------------------------ setup
    def next_key(self):
        self.key, key = jax.random.split(self.key)
        return key

    def build_net(self, direction: str):
        net = build_model(self.model_config, self.next_key())
        self.nets[direction] = net
        self.ema[direction] = eqx.filter(net, eqx.is_inexact_array)

    def split(self, direction: str):
        return eqx.partition(self.nets[direction], eqx.is_inexact_array)

    def ema_net(self, direction: str):
        params, static = self.split(direction)
        return eqx.combine(self.ema[direction] if self.config.ema else params, static)

    def load_checkpoints(self):
        c = self.config
        for direction, path, sample_path in (("f", c.checkpoint_f, c.sample_checkpoint_f), ("b", c.checkpoint_b, c.sample_checkpoint_b)):
            if path:
                self.nets[direction] = eqx.tree_deserialise_leaves(path, self.nets[direction])
                self.ema[direction] = eqx.filter(self.nets[direction], eqx.is_inexact_array)
            if sample_path:
                sample_net = eqx.tree_deserialise_leaves(sample_path, self.nets[direction])
                self.ema[direction] = eqx.filter(sample_net, eqx.is_inexact_array)
            print(f"loaded {direction}: net={path or '-'} ema={sample_path or '-'}", flush=True)

    def _compile(self):
        config, T = self.config, self.T
        mean_match, ema_rate = config.mean_match, config.ema_rate
        optimizer = self.optimizer
        gammas, time = self.gammas, self.time
        mean_final, var_final = self.mean_final, self.var_final

        def batched(params, static):
            net = eqx.combine(params, static)
            return lambda xs, ts: jax.vmap(net)(xs, ts)

        @eqx.filter_jit
        def train_step(params, static, opt_state, ema_params, cache, permutation, iteration):
            xs, outs, steps = minibatch(cache, permutation, iteration, config.batch_size)

            def loss_fn(p):
                pred = batched(p, static)(xs, T - steps)
                if mean_match:
                    pred = pred - xs
                return jnp.mean(jnp.square(pred - outs))

            loss, grads = jax.value_and_grad(loss_fn)(params)
            grad_norm = optax.global_norm(grads)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = eqx.apply_updates(params, updates)
            ema_params = jax.tree.map(lambda e, p: (1.0 - ema_rate) * p + ema_rate * e, ema_params, params)
            return params, opt_state, ema_params, loss, grad_norm

        @eqx.filter_jit
        def cache_init(key, x0):
            return flatten_trajectories(*record_init_langevin(key, x0, gammas, time, mean_final, var_final))

        @eqx.filter_jit
        def cache_from_net(key, params, static, x_init):
            trajectories = record_langevin_seq(key, batched(params, static), x_init, gammas, time, mean_match)
            return flatten_trajectories(*trajectories)

        @eqx.filter_jit
        def sample(key, params, static, x_init):
            x_tot, _, _ = record_langevin_seq(key, batched(params, static), x_init, gammas, time, mean_match, sample=True)
            return x_tot

        @functools.partial(jax.jit, static_argnames=("batch_size",))
        def data_batch(key, images, batch_size):
            return sample_images(key, images, batch_size, config.random_flip)

        @functools.partial(jax.jit, static_argnames=("batch_size",))
        def prior_batch(key, batch_size):
            return mean_final + jnp.sqrt(var_final) * jax.random.normal(key, (batch_size,) + self.shape)

        @jax.jit
        def init_trajectories(key, x0):
            return record_init_langevin(key, x0, gammas, time, mean_final, var_final)[0]

        self._train_step = train_step
        self._cache_init = cache_init
        self._cache_from_net = cache_from_net
        self._sample = sample
        self._data_batch = lambda key, batch_size: data_batch(key, self.images, batch_size=batch_size)
        self._prior_batch = lambda key, batch_size: prior_batch(key, batch_size=batch_size)
        self._init_trajectories = init_trajectories

    # ------------------------------------------------------------------ cache
    def new_cache(self, direction: str, n: int) -> Cache:
        config = self.config
        sample_direction = "f" if direction == "b" else "b"
        sample_params, sample_static = eqx.partition(self.ema_net(sample_direction), eqx.is_inexact_array)
        parts = []
        start = time.time()
        for _ in range(config.num_cache_batches):
            batch_key, chain_key = jax.random.split(self.next_key())
            if direction == "b":
                x0 = self._data_batch(batch_key, config.cache_npar)
                if n == 1:
                    parts.append(self._cache_init(chain_key, x0))
                else:
                    parts.append(self._cache_from_net(chain_key, sample_params, sample_static, x0))
            else:
                x_init = self._prior_batch(batch_key, config.cache_npar)
                parts.append(self._cache_from_net(chain_key, sample_params, sample_static, x_init))
        cache = concatenate_caches(parts) if len(parts) > 1 else parts[0]
        jax.block_until_ready(cache)
        print(f"cache {direction} n={n}: rows={cache.xs.shape[0]} built in {time.time() - start:.1f}s", flush=True)
        return cache

    # ------------------------------------------------------------------ one half-step
    def ipf_step(self, direction: str, n: int):
        config = self.config
        cache = self.new_cache(direction, n)
        if not config.use_prev_net:
            self.build_net(direction)
        params, static = self.split(direction)
        ema_params = self.ema[direction]
        opt_state = self.optimizer.init(params)
        permutation = jax.random.permutation(self.next_key(), cache.xs.shape[0])

        start = time.time()
        losses = []
        for i in range(config.num_iter + 1):
            if i % config.cache_refresh_stride == 0 and i > 0:
                self.nets[direction] = eqx.combine(params, static)
                self.ema[direction] = ema_params
                cache = None
                cache = self.new_cache(direction, n)
                permutation = jax.random.permutation(self.next_key(), cache.xs.shape[0])
            params, opt_state, ema_params, loss, grad_norm = self._train_step(
                params, static, opt_state, ema_params, cache, permutation, jnp.int32(i % config.cache_refresh_stride))
            losses.append(loss)
            if i % config.log_stride == 0 and i > 0:
                loss_value, grad_value = float(loss), float(grad_norm)
                if not np.isfinite(loss_value):
                    raise FloatingPointError(f"non-finite loss at {direction} n={n} i={i}")
                self.logger.log(fb=direction, n=n, i=i, global_step=i + config.num_iter * n, loss=loss_value,
                                grad_norm=grad_value, elapsed=round(time.time() - start, 2))
                if i % (config.log_stride * 20) == 0:
                    print(f"{direction} n={n} i={i}/{config.num_iter} loss={loss_value:.5f} grad={grad_value:.3f} "
                          f"{i / (time.time() - start):.1f} it/s", flush=True)
            self.nets[direction] = eqx.combine(params, static)
            self.ema[direction] = ema_params
            self.save_step(i, n, direction)
        self.nets[direction] = eqx.combine(params, static)
        self.ema[direction] = ema_params
        mean_loss = float(jnp.mean(jnp.stack(losses[-max(1, len(losses) // 10):])))
        print(f"half-step {direction} n={n} done in {time.time() - start:.1f}s, last-10% mean loss {mean_loss:.5f}", flush=True)
        return mean_loss

    # ------------------------------------------------------------------ checkpoints and samples
    def save_step(self, i: int, n: int, direction: str):
        config = self.config
        due = (i % config.gif_stride == 0 and i > 0) or i == config.num_iter
        if not due:
            return
        tag = f"{direction}_{n}_{i}"
        eqx.tree_serialise_leaves(os.path.join(self.out_dir, "checkpoints", f"net_{tag}.eqx"), self.nets[direction])
        if config.ema:
            eqx.tree_serialise_leaves(os.path.join(self.out_dir, "checkpoints", f"sample_net_{tag}.eqx"), self.ema_net(direction))
        self.plot_samples(i, n, direction)

    def plot_samples(self, i: int, n: int, direction: str):
        config = self.config
        batch_key, chain_key = jax.random.split(jax.random.PRNGKey(0))
        if direction == "f":
            batch = self._data_batch(batch_key, config.plot_npar)
        else:
            batch = self._prior_batch(batch_key, config.plot_npar)
        params, static = eqx.partition(self.ema_net(direction), eqx.is_inexact_array)
        x_tot = self._sample(chain_key, params, static, batch)
        self.write_grid(batch, x_tot, i, n, direction)

    def write_grid(self, batch, x_tot, i, n, direction):
        init_x, final_x = np.asarray(batch), np.asarray(x_tot[-1])
        directory = os.path.join(self.out_dir, "im", f"{direction}_{n}_{i}")
        save_image_grid(init_x, os.path.join(directory, "im_grid_first.png"))
        save_image_grid(final_x, os.path.join(directory, "im_grid_final.png"))
        self.plot_logger.log(fb=direction, n=n, i=i, init_var=float(init_x.var()), final_var=float(final_x.var()),
                             mean_init=float(init_x.mean()), mean_final=float(final_x.mean()), T=self.T)
        print(f"samples {direction} n={n} i={i}: init var {init_x.var():.4f} final var {final_x.var():.4f} "
              f"final mean {final_x.mean():.4f}", flush=True)

    # ------------------------------------------------------------------ outer loop
    def train(self):
        config = self.config
        batch_key, chain_key = jax.random.split(jax.random.PRNGKey(0))
        init_sample = self._data_batch(batch_key, config.plot_npar)
        x_tot = self._init_trajectories(chain_key, init_sample)
        self.write_grid(init_sample, x_tot, 0, 0, "f")

        start_it = config.checkpoint_it if config.checkpoint_run else 1
        start_pass = config.checkpoint_pass if config.checkpoint_run else "b"
        for n in range(start_it, config.n_ipf + 1):
            print(f"IPF iteration {n}/{config.n_ipf}", flush=True)
            if start_pass == "f" and n == start_it:
                self.ipf_step("f", n)
            else:
                self.ipf_step("b", n)
                self.ipf_step("f", n)
