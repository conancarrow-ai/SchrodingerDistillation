"""Image denoisers assembled from unmodified sparse-transformers blocks.

A model maps an image x in R^{C x H x W} and a time t to R^{C x H x W}. The image
is cut into non-overlapping p x p patches, each patch is embedded into R^D by a
(sparse) linear layer, and the resulting token sequence is processed by the
blocks of either family of the sparse-transformers code:

  st  : causal softmax attention with an optional local window, sparse linears
        with fan-in K, RMS normalisation (st.transformer.Block);
  z1t : attention-free attention (conv, recurrence or full), sparse linears with
        fan-in K, dynamic tanh normalisation (z1t.components.Block). The z1t
        blocks take their linear layers from a factory; by default they get the
        masked sparse linear of st (sparse_linear="mask"), which computes the
        same map as z1t's gather layer but is far faster on GPU at these widths.

Both block families are causal, so a token only sees the tokens before it. To
give every output patch the whole image as context without touching the blocks,
the sequence is [time token, image tokens, image tokens] and the outputs are
read from the second copy (context_prefix=True). The time t enters through a
sinusoidal embedding and a two-layer MLP, as the first token and as an additive
shift of every image token.

With residual_output=True the model returns x + f(x, t), the mean-matching
convention of the reference code in which the network output is the mean of the
next state; the output projection is zero-initialised so the model starts as the
identity.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from st.transformer import Block as SoftmaxBlock, SparseLinear as MaskedSparseLinear, sparse_count
from z1t.components import Block as AFTBlock, DyT, SparseLinear as GatherSparseLinear, attention_factory, linear_factory

FAMILIES = ("st", "z1t")


@dataclass
class ModelConfig:
    family: str = "st"
    channels: int = 3
    image_size: int = 64
    patch_size: int = 4
    n_layers: int = 6
    n_embed: int = 256
    n_head: int = 4                    # attention heads (st) or AFT-conv heads (z1t)
    attn_window: int | None = None     # st only; None is dense causal attention
    linear_fan_in: int | None = None   # K; None or at least the input width gives a dense layer
    sparse_linear: str = "mask"        # sparse layer implementation: "mask" (st: dense weight under a fixed
                                       # boolean mask) or "gather" (z1t: (n_out, K) weights and gather indices;
                                       # 20x slower backward pass on GPU at these widths)
    aft_kind: str = "conv"             # z1t only: conv, recurrence or full
    aft_ksize: int = 4                 # z1t only
    dyt_alpha: float = 0.5             # z1t only
    context_prefix: bool = True
    residual_output: bool = True
    time_embed_dim: int = 64
    remat: bool = False

    @property
    def tokens(self) -> int:
        if self.image_size % self.patch_size:
            raise ValueError("patch_size must divide image_size")
        return (self.image_size // self.patch_size) ** 2

    @property
    def patch_dim(self) -> int:
        return self.channels * self.patch_size ** 2

    @property
    def sequence(self) -> int:
        return 1 + (2 if self.context_prefix else 1) * self.tokens


def sinusoidal_embedding(t, dim: int, max_period: float = 10000.0):
    """The timestep embedding of the reference U-Net: [cos(t w_i), sin(t w_i)], w_i = max_period^(-i / half)."""
    half = dim // 2
    frequencies = jnp.exp(-math.log(max_period) * jnp.arange(half, dtype=jnp.float32) / half)
    angles = t.astype(jnp.float32) * frequencies
    embedding = jnp.concatenate([jnp.cos(angles), jnp.sin(angles)])
    if dim % 2:
        embedding = jnp.concatenate([embedding, jnp.zeros((1,), embedding.dtype)])
    return embedding


SPARSE_LINEARS = ("mask", "gather")


def make_linear_factory(fan_in: int | None, sparse_linear: str = "mask"):
    """A constructor (ins, outs, key) -> linear layer with fan-in K: dense when K reaches the input width,
    otherwise the masked (st) or the gather (z1t) sparse linear layer. Both compute the same map."""
    if sparse_linear == "mask":
        def make(ins, outs, key):
            k = sparse_count(fan_in, ins)
            if k >= ins:
                return eqx.nn.Linear(ins, outs, key=key)
            return MaskedSparseLinear(ins, outs, k, key=key)
        return make
    if sparse_linear == "gather":
        factory = linear_factory(fan_in)
        return lambda ins, outs, key: factory(ins, outs, key=key)
    raise ValueError(f"unknown sparse_linear {sparse_linear!r} (expected one of {SPARSE_LINEARS})")


def zero_linear(layer):
    """Zero the weights and bias of a dense or sparse linear layer of either family."""
    if isinstance(layer, MaskedSparseLinear):
        return eqx.tree_at(lambda l: l.linear, layer, zero_linear(layer.linear))
    if isinstance(layer, (eqx.nn.Linear, GatherSparseLinear)):
        layer = eqx.tree_at(lambda l: l.weight, layer, jnp.zeros_like(layer.weight))
        if layer.bias is not None:
            layer = eqx.tree_at(lambda l: l.bias, layer, jnp.zeros_like(layer.bias))
        return layer
    raise TypeError(type(layer))


def patchify(x: Float[Array, "C H W"], patch_size: int) -> Float[Array, "T P"]:
    channels, height, width = x.shape
    rows, cols = height // patch_size, width // patch_size
    x = x.reshape(channels, rows, patch_size, cols, patch_size)
    x = jnp.transpose(x, (1, 3, 0, 2, 4))
    return x.reshape(rows * cols, channels * patch_size * patch_size)


def unpatchify(tokens: Float[Array, "T P"], channels: int, height: int, width: int, patch_size: int) -> Float[Array, "C H W"]:
    rows, cols = height // patch_size, width // patch_size
    x = tokens.reshape(rows, cols, channels, patch_size, patch_size)
    x = jnp.transpose(x, (2, 0, 3, 1, 4))
    return x.reshape(channels, height, width)


class TimeMLP(eqx.Module):
    linear1: eqx.Module
    linear2: eqx.Module
    embed_dim: int = eqx.field(static=True)

    def __init__(self, embed_dim: int, n_embed: int, make_linear, key):
        key1, key2 = jax.random.split(key)
        self.embed_dim = embed_dim
        self.linear1 = make_linear(embed_dim, n_embed, key1)
        self.linear2 = make_linear(n_embed, n_embed, key2)

    def __call__(self, t):
        h = jax.nn.silu(self.linear1(sinusoidal_embedding(t, self.embed_dim)))
        return self.linear2(h)


class SparseTransformerDenoiser(eqx.Module):
    embed: eqx.Module
    unembed: eqx.Module
    position: eqx.nn.Embedding
    time_mlp: TimeMLP
    blocks: list
    norm: eqx.Module
    family: str = eqx.field(static=True)
    channels: int = eqx.field(static=True)
    image_size: int = eqx.field(static=True)
    patch_size: int = eqx.field(static=True)
    tokens: int = eqx.field(static=True)
    context_prefix: bool = eqx.field(static=True)
    residual_output: bool = eqx.field(static=True)
    remat: bool = eqx.field(static=True)

    def __init__(self, config: ModelConfig, key):
        if config.family not in FAMILIES:
            raise ValueError(f"unknown family {config.family!r} (expected one of {FAMILIES})")
        keys = jax.random.split(key, 5 + config.n_layers)
        make_linear = make_linear_factory(config.linear_fan_in, config.sparse_linear)
        block_linear = lambda ins, outs, *, key: make_linear(ins, outs, key)
        dim, sequence = config.n_embed, config.sequence

        self.family = config.family
        self.channels, self.image_size, self.patch_size = config.channels, config.image_size, config.patch_size
        self.tokens = config.tokens
        self.context_prefix, self.residual_output, self.remat = config.context_prefix, config.residual_output, config.remat

        self.embed = make_linear(config.patch_dim, dim, keys[0])
        self.unembed = zero_linear(make_linear(dim, config.patch_dim, keys[1]))
        self.position = eqx.nn.Embedding(sequence, dim, key=keys[2])
        self.time_mlp = TimeMLP(config.time_embed_dim, dim, make_linear, keys[3])
        block_keys = keys[4:4 + config.n_layers]
        if config.family == "st":
            self.blocks = [
                SoftmaxBlock(dim, config.n_head, sequence, config.attn_window, config.linear_fan_in, k)
                for k in block_keys
            ]
            self.norm = eqx.nn.RMSNorm(dim)
        else:
            attention = attention_factory(
                config.aft_kind, max_seq=sequence, heads=config.n_head, ksize=config.aft_ksize,
                linear=block_linear,
            )
            self.blocks = [
                AFTBlock(dim, key=k, attn=attention, mlp_linear=block_linear,
                         mlp_act=jax.nn.silu, dyt_alpha=config.dyt_alpha)
                for k in block_keys
            ]
            self.norm = DyT(dim, alpha_init=config.dyt_alpha)

    def _run_block(self, block, h):
        if self.family == "st":
            return block(h, None)
        return block(h)

    def __call__(self, x: Float[Array, "C H W"], t) -> Float[Array, "C H W"]:
        tokens = patchify(x, self.patch_size)
        h = jax.vmap(self.embed)(tokens)
        e = self.time_mlp(jnp.asarray(t))
        h = h + e[None, :]
        parts = [e[None, :], h, h] if self.context_prefix else [e[None, :], h]
        sequence = jnp.concatenate(parts, axis=0)
        sequence = sequence + jax.vmap(self.position)(jnp.arange(sequence.shape[0]))

        run = eqx.filter_checkpoint(self._run_block) if self.remat else self._run_block
        for block in self.blocks:
            sequence = run(block, sequence)

        out = sequence[-self.tokens:]
        out = jax.vmap(self.norm)(out)
        out = jax.vmap(self.unembed)(out)
        f = unpatchify(out, self.channels, self.image_size, self.image_size, self.patch_size)
        return x + f if self.residual_output else f


def build_model(config: ModelConfig, key) -> SparseTransformerDenoiser:
    return SparseTransformerDenoiser(config, key)


def parameter_counts(model) -> tuple[int, int]:
    """(stored trainable floats, active parameters): a masked sparse linear stores a dense
    weight but only its K-sparse support is active; a gather sparse linear stores only the support."""

    def is_sparse(node):
        return isinstance(node, (MaskedSparseLinear, GatherSparseLinear))

    stored = sum(int(leaf.size) for leaf in jax.tree.leaves(eqx.filter(model, eqx.is_inexact_array)))
    active = 0
    for node in jax.tree.leaves(model, is_leaf=is_sparse):
        if isinstance(node, MaskedSparseLinear):
            active += int(node.mask.sum()) + (0 if node.linear.bias is None else int(node.linear.bias.size))
        elif isinstance(node, GatherSparseLinear):
            active += int(node.weight.size) + int(node.bias.size)
        elif eqx.is_inexact_array(node):
            active += int(node.size)
    return stored, active
