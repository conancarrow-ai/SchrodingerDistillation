"""The toy two-dimensional distributions and the score network of the dsb reference code (Model: Basic).

Distributions reproduce bridge/data/two_dim.py without scikit-learn: moon (two
interleaving half circles), scurve (the x and z coordinates of an S curve), circle
(two concentric circles, factor 0.5, scaled by 10) and mixture (two unit Gaussians
at x = -7 and x = 7). moon and scurve are standardised by the mean and the unbiased
standard deviation of all their entries and scaled by 7. The reference draws npar
points once and keeps them; its prior is N(mean_final, var_final I) with mean_final
the mean of the first 100 points and var_final the unbiased variance of their
entries (final_adaptive). The first half of the mixture points is the left
Gaussian, so that prior is centred on the left mode, as in the reference.

ScoreNetwork(x, t): the encoders 2 -> 16 -> 32 (of x) and 16 -> 16 -> 32 (of the
Fairseq sinusoidal embedding of t) are concatenated and decoded by
64 -> 128 -> 128 -> 2, with leaky ReLU (slope 0.01) between the layers. Each
linear layer is dense (the reference) or a sparse linear layer of the
sparse-transformers code with fan-in kappa: st's masked layer (mask) or z1t's
gather layer (gather); the fan-in is clamped to the input width of each layer.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from st.transformer import SparseLinear as MaskedSparseLinear, sparse_count
from z1t.components import SparseLinear as GatherSparseLinear

DISTRIBUTIONS = ("moon", "scurve", "circle", "mixture")
LINEAR_KINDS = ("dense", "mask", "gather")
PRIOR_COUNT = 100
PLOT_LIMIT = 15.0


# ---------------------------------------------------------------- distributions
def standardise(points: np.ndarray) -> np.ndarray:
    return (points - points.mean()) / points.std(ddof=1)


def toy_points(distribution: str, count: int, seed: int) -> np.ndarray:
    """count points of the named distribution as float32 (count, 2), in the storage order of the reference."""
    generator = np.random.default_rng(seed)
    if distribution == "moon":
        outer, inner = count // 2, count - count // 2
        angle_outer, angle_inner = np.linspace(0, np.pi, outer), np.linspace(0, np.pi, inner)
        points = np.concatenate([np.stack([np.cos(angle_outer), np.sin(angle_outer)], 1),
                                 np.stack([1 - np.cos(angle_inner), 1 - np.sin(angle_inner) - 0.5], 1)])
        points = standardise(points[generator.permutation(count)]) * 7.0
    elif distribution == "scurve":
        angle = 3 * np.pi * (generator.uniform(size=count) - 0.5)
        points = np.stack([np.sin(angle), np.sign(angle) * (np.cos(angle) - 1)], 1)
        points = standardise(points) * 7.0
    elif distribution == "circle":
        outer, inner = count // 2, count - count // 2
        angle_outer = np.linspace(0, 2 * np.pi, outer, endpoint=False)
        angle_inner = np.linspace(0, 2 * np.pi, inner, endpoint=False)
        points = np.concatenate([np.stack([np.cos(angle_outer), np.sin(angle_outer)], 1),
                                 0.5 * np.stack([np.cos(angle_inner), np.sin(angle_inner)], 1)])
        points = points[generator.permutation(count)] * 10.0
    elif distribution == "mixture":
        points = generator.standard_normal((count, 2))
        points[: count // 2, 0] -= 7.0
        points[count // 2 :, 0] += 7.0
    else:
        raise ValueError(f"unknown distribution {distribution!r} (expected one of {DISTRIBUTIONS})")
    return np.asarray(points, dtype=np.float32)


def toy_prior(points: np.ndarray, count: int = PRIOR_COUNT) -> tuple[np.ndarray, float]:
    """(mean_final, var_final) of the final_adaptive prior: the mean of the first `count` points per
    coordinate and the unbiased variance of their entries."""
    first = np.asarray(points[:count], dtype=np.float64)
    return first.mean(axis=0).astype(np.float32), float(first.var(ddof=1))


def toy_targets(points: np.ndarray, mean_final, var_final: float, count: int) -> tuple[np.ndarray, np.ndarray]:
    """(data_target, prior_target): `count` data points and `count` prior samples, drawn once with a fixed
    generator, against which the samples of a run are measured and drawn."""
    generator = np.random.default_rng(1)
    count = min(count, points.shape[0])
    data_target = np.asarray(points, np.float32)[generator.choice(points.shape[0], count, replace=False)]
    prior_target = (np.asarray(mean_final, np.float32) + np.sqrt(var_final) * generator.standard_normal((count, points.shape[1]))).astype(np.float32)
    return data_target, prior_target


# ---------------------------------------------------------------- model
@dataclass
class ToyModelConfig:
    encoder_layers: list = field(default_factory=lambda: [16])
    pos_dim: int = 16
    decoder_layers: list = field(default_factory=lambda: [128, 128])
    x_dim: int = 2
    sparse_linear: str = "dense"       # dense (the reference), mask (st) or gather (z1t)
    linear_fan_in: int | None = None   # kappa of the sparse layers; None is the input width
    swap_init: bool = False            # give each sparse layer the initial scale of the other implementation

    @property
    def label(self) -> str:
        if self.sparse_linear == "dense":
            return "dense"
        label = f"{self.sparse_linear} K={self.linear_fan_in if self.linear_fan_in is not None else 'full'}"
        return label + (" (swapped init)" if self.swap_init else "")


def fairseq_embedding(t, dim: int):
    """The sinusoidal embedding of bridge/models/basic/time_embedding.py: [sin(t w_i), cos(t w_i)],
    w_i = 10000^(-i / (half - 1)), half = dim // 2."""
    half = dim // 2
    frequencies = jnp.exp(-math.log(10000.0) * jnp.arange(half, dtype=jnp.float32) / (half - 1))
    angles = jnp.asarray(t, jnp.float32) * frequencies
    embedding = jnp.concatenate([jnp.sin(angles), jnp.cos(angles)])
    if dim % 2:
        embedding = jnp.concatenate([embedding, jnp.zeros((1,), embedding.dtype)])
    return embedding


def make_toy_linear(sparse_linear: str, fan_in: int | None, swap_init: bool = False):
    """A constructor (ins, outs, key) -> layer. The mask and gather kinds always build the sparse-transformers
    class, with the fan-in clamped to [1, ins]; a full fan-in gives a mask of ones.

    The two classes initialise differently: the masked layer keeps the dense U(-1/sqrt(ins), 1/sqrt(ins))
    weights and bias of eqx.nn.Linear under its mask, the gather layer draws its K weights per output from
    U(-1/sqrt(K), 1/sqrt(K)) with a zero bias. With swap_init each gets the other's initial scale."""
    if sparse_linear == "dense":
        return lambda ins, outs, key: eqx.nn.Linear(ins, outs, key=key)
    if sparse_linear == "mask":
        def make_mask(ins, outs, key):
            k = sparse_count(fan_in, ins)
            layer = MaskedSparseLinear(ins, outs, k, key=key)
            if swap_init:
                layer = eqx.tree_at(lambda l: l.linear.weight, layer, layer.linear.weight * math.sqrt(ins / k))
                layer = eqx.tree_at(lambda l: l.linear.bias, layer, jnp.zeros_like(layer.linear.bias))
            return layer
        return make_mask
    if sparse_linear == "gather":
        def make_gather(ins, outs, key):
            k = sparse_count(fan_in, ins)
            layer = GatherSparseLinear(ins, outs, k=k, key=key)
            if swap_init:
                bound = 1.0 / math.sqrt(ins)
                layer = eqx.tree_at(lambda l: l.weight, layer, layer.weight * math.sqrt(k / ins))
                layer = eqx.tree_at(lambda l: l.bias, layer, jax.random.uniform(jax.random.fold_in(key, 1), (outs,), minval=-bound, maxval=bound))
            return layer
        return make_gather
    raise ValueError(f"unknown sparse_linear {sparse_linear!r} (expected one of {LINEAR_KINDS})")


class MLP(eqx.Module):
    layers: list

    def __init__(self, input_dim: int, widths, make_linear, key):
        keys = jax.random.split(key, len(widths))
        dims = [input_dim] + list(widths)
        self.layers = [make_linear(dims[index], dims[index + 1], keys[index]) for index in range(len(widths))]

    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = jax.nn.leaky_relu(layer(x), 0.01)
        return self.layers[-1](x)


class ScoreNetwork(eqx.Module):
    net: MLP
    t_encoder: MLP
    x_encoder: MLP
    pos_dim: int = eqx.field(static=True)

    def __init__(self, config: ToyModelConfig, key):
        key_net, key_time, key_x = jax.random.split(key, 3)
        make_linear = make_toy_linear(config.sparse_linear, config.linear_fan_in, config.swap_init)
        t_enc_dim = 2 * config.pos_dim
        self.pos_dim = config.pos_dim
        self.net = MLP(2 * t_enc_dim, list(config.decoder_layers) + [config.x_dim], make_linear, key_net)
        self.t_encoder = MLP(config.pos_dim, list(config.encoder_layers) + [t_enc_dim], make_linear, key_time)
        self.x_encoder = MLP(config.x_dim, list(config.encoder_layers) + [t_enc_dim], make_linear, key_x)

    def __call__(self, x, t):
        temb = self.t_encoder(fairseq_embedding(t, self.pos_dim))
        xemb = self.x_encoder(x)
        return self.net(jnp.concatenate([xemb, temb]))


# ---------------------------------------------------------------- validation metrics and figures
def energy_distance(a: np.ndarray, b: np.ndarray) -> float:
    """2 E|X - Y| - E|X - X'| - E|Y - Y'| between two point clouds (zero iff the distributions agree)."""

    def mean_distance(x, y):
        return float(np.sqrt(((x[:, None, :] - y[None, :, :]) ** 2).sum(-1)).mean())

    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return 2 * mean_distance(a, b) - mean_distance(a, a) - mean_distance(b, b)


def _axes_limits(axis):
    axis.set_xlim(-PLOT_LIMIT, PLOT_LIMIT)
    axis.set_ylim(-PLOT_LIMIT, PLOT_LIMIT)
    axis.set_aspect("equal")
    axis.set_xticks([])
    axis.set_yticks([])


def draw_samples(path: str, final, target, title: str, target_label: str):
    """Final samples over the target cloud, one square panel."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(3.2, 3.2))
    axis.scatter(target[:, 0], target[:, 1], s=3, color="0.75", label=target_label)
    axis.scatter(final[:, 0], final[:, 1], s=3, color="tab:blue", label="samples")
    _axes_limits(axis)
    axis.set_title(title, fontsize=9)
    axis.legend(loc="lower left", fontsize=7, markerscale=3, frameon=False)
    figure.tight_layout(pad=0.3)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    figure.savefig(path, dpi=110)
    plt.close(figure)


def draw_trajectories(path: str, x_tot, start, title: str, count: int = 12):
    """The chains of `count` particles over the start and end clouds."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(3.2, 3.2))
    axis.scatter(start[:, 0], start[:, 1], s=3, color="0.85")
    axis.scatter(x_tot[-1, :, 0], x_tot[-1, :, 1], s=3, color="tab:blue", alpha=0.6)
    for particle in range(min(count, x_tot.shape[1])):
        chain = np.concatenate([start[particle][None], x_tot[:, particle]])
        axis.plot(chain[:, 0], chain[:, 1], color="tab:green", lw=1)
        axis.plot(chain[0, 0], chain[0, 1], "rx", ms=4)
        axis.plot(chain[-1, 0], chain[-1, 1], "r.", ms=4)
    _axes_limits(axis)
    axis.set_title(title, fontsize=9)
    figure.tight_layout(pad=0.3)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    figure.savefig(path, dpi=110)
    plt.close(figure)


def draw_points(path: str, points, mean_final, var_final, title: str):
    """The training points with the prior's mean and one-sigma circle."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(3.2, 3.2))
    axis.scatter(points[:, 0], points[:, 1], s=3, color="tab:gray")
    circle = plt.Circle((float(mean_final[0]), float(mean_final[1])), math.sqrt(var_final), fill=False, color="tab:orange", lw=1)
    axis.add_patch(circle)
    axis.plot(float(mean_final[0]), float(mean_final[1]), "o", color="tab:orange", ms=3)
    _axes_limits(axis)
    axis.set_title(title, fontsize=9)
    figure.tight_layout(pad=0.3)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    figure.savefig(path, dpi=110)
    plt.close(figure)
