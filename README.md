# Schrödinger distillation: IPF on sparse-transformers models

Two upstream projects are vendored as submodules:

- `diffusion_schrodinger_bridge/` (dsb): the PyTorch reference implementation of
  Diffusion Schrödinger Bridge. The object of interest is
  `IPFSequential.ipf_step` in `bridge/runners/ipf.py`, with `bridge/langevin.py`
  and `bridge/data/cacheloader.py`.
- `sparse-transformers/` (st): JAX/Equinox transformer blocks with fixed-sparsity
  linear layers (`research/st`) and attention-free (AFT) blocks with dynamic tanh
  normalisation (`research/z1t`).

Deliverables:

1. `report/unet_celeba.pdf` (source `report/unet_celeba.tex`): one-page block
   diagram of the U-Net the dsb code instantiates for CelebA 64x64, in
   mathematical notation, with a table of every component, its domain and
   codomain, its parameter count and a colour code that says which components
   exist as objects of the sparse-transformers classes, which could be made
   sparse with an adapter, and which could be substituted.
2. `sparse_dsb/`: a JAX port of the dsb IPF routines that trains unmodified
   sparse-transformers blocks as the forward and backward networks of the bridge.
3. A CelebA 64x64 smoke test (section Results).

## Which way round: port dsb to JAX, not st to PyTorch

The two options in goal.txt were (a) re-implement the sparse-transformers
models as `nn.Module`s for the dsb code, or (b) re-implement the dsb routines
in JAX and leave the sparse-transformers code untouched. (b) was chosen.

- Amount of work. The IPF logic is 250 lines of PyTorch (`ipf_step`, the two
  trajectory recorders, the cache, the EMA helper); the sparse-transformers code
  is 900 lines spread over two packages, each with its own sparse linear layer,
  three attention variants, normalisation layers and parameter accounting. Porting
  the smaller side duplicates less.
- Fidelity to st. Option (a) would have produced a second implementation of
  every st object, and any experiment on "models like those found in
  sparse-transformers" would have run on the copy. With (b) the blocks are the
  st and z1t classes themselves, imported from the submodule.
- Performance. The reference loop runs the diffusion chain as a Python loop of
  PyTorch calls and feeds the trajectory cache back through a `DataLoader`. In
  JAX the chain is one `lax.scan`, the whole cache lives on the device, and
  sampling a minibatch, the gradient step, the Adam update and the EMA update are
  one jitted program. The smoke test below runs at roughly 15 gradient steps per
  second with a 513-token sequence per image on one GPU, which is the speed of
  the training step itself.
- The st and z1t training scripts already use the `eqx.filter_jit` + `optax`
  pattern, so the port composes with them (the sparse linear masks and gather
  indices are non-float leaves and are excluded from the optimiser and the EMA
  by the same `eqx.is_inexact_array` partition the st code uses).

## The port (`sparse_dsb/`)

| dsb (PyTorch) | here | notes |
|---|---|---|
| `IPFBase.__init__` gamma schedule, `T` | `schedule.py` | linear or geometric ramp mirrored; `time[k] = sum_{j<=k} gamma_j`, `T = time[-1]` |
| `Langevin.record_init_langevin` | `langevin.record_init_langevin` | OU reference chain from data, `out = t_old - t_new`, one `lax.scan` |
| `Langevin.record_langevin_seq` | `langevin.record_langevin_seq` | chain driven by a network; `mean_match` and `sample` flags as upstream |
| `CacheLoader` | `cache.py` | rows `(x, out, t)` with the same flattening; lives on the device |
| `IPFSequential.ipf_step` | `ipf.IPF.ipf_step` | cache refresh every `cache_refresh_stride`, loss `mean |net(x, T - t) - x - out|^2`, Adam + global-norm clipping, EMA update, fresh optimiser per half-step, `use_prev_net` |
| `IPFSequential.train` | `ipf.IPF.train` | initial reference pass plot, then `b` and `f` half-steps for `n = 1..n_ipf`, `checkpoint_it` / `checkpoint_pass` resume |
| `save_step` | `ipf.IPF.save_step` | checkpoints `net_{fb}_{n}_{i}.eqx` and `sample_net_...eqx` (EMA), sample grids `im/{fb}_{n}_{i}/im_grid_{first,final}.png`, variance log |
| `get_datasets` (CelebA) | `data.py` | center crop 140, resize 64, random horizontal flip, adaptive prior mean over the first 100 images, `var_final = 0.5` |

Conventions kept from the reference code, including its quirks:

- The networks are trained with the input `T - time[k]` and simulated with the
  input `time[k]`. For the symmetric schedule the two differ by one step size at
  the matching position; the port keeps the reference behaviour.
- The `time_sampler` / `weight_distrib` options of the reference code are
  constructed but never used by its image runs, so they are not ported;
  `fast_sampling` and `n_ipf_init` are likewise dead options upstream.
- The time input to a network is a float (cumulative time), fed to a sinusoidal
  embedding exactly as the U-Net does.

Differences:

- Cache rows are visited in a random permutation drawn at every refresh. The
  reference code iterates the cache in storage order (consecutive steps of the
  same trajectory in one minibatch); with `batch_size x cache_refresh_stride`
  smaller than the cache (the CelebA setting) both visit each row at most once per
  refresh.
- The EMA copy of the opposite network is rebuilt from its parameter tree instead
  of re-instantiating the module from saved constructor arguments.
- Single device. Data parallelism over the GPUs of one node is a sharding
  annotation away but is not implemented.

### Models (`sparse_dsb/models.py`)

`SparseTransformerDenoiser` maps an image `x in R^{C x H x W}` and a time `t` to
`R^{C x H x W}`:

1. cut `x` into `p x p` patches, `T = (H/p)(W/p)` tokens of dimension `C p^2`;
2. embed each token with a linear layer of the chosen family (sparse with fan-in
   `K` when `K` is below the input width, dense otherwise);
3. embed `t` sinusoidally and through a two-layer MLP into `e in R^D`; add `e`
   to every token and prepend it as a token;
4. run the sequence `[e, tokens, tokens]` through `n_layers` unmodified blocks:
   `st.transformer.Block` (causal softmax attention with optional local window,
   `SparseLinear` projections, RMS norm) or `z1t.components.Block` (AFT conv /
   recurrence / full attention, `SparseLinear` projections, DyT);
5. read the last `T` outputs, normalise, project back to patches with a
   zero-initialised linear layer, reassemble the image, return `x + f(x, t)`.

Both block families are causal. The duplicated token sequence
(`context_prefix: true`) gives every output patch the whole image as context
without modifying the blocks: the first copy is the context, the second copy
produces the outputs. The test suite checks this (perturbing the last patch moves
the first output patch with the prefix and leaves it unchanged without).

The zero-initialised output projection makes every network start as the
identity map, which is the right starting point for the mean-matching
parametrisation of the reference code (`pred = net(x, t) - x`).

### Fan-in

`fan_in` of `SparseLinear` is written `K`. Each output unit of a sparse linear
layer `R^{n_in} -> R^{n_out}` has exactly `K` inputs drawn uniformly without
replacement (both implementations; st stores a dense weight under a boolean
mask, z1t stores the `(n_out, K)` weights and gather indices). `K` is therefore
an exact in-degree, not the maximum degree `K_G = max_v k_v` of the network
graph: an input unit has degree equal to its out-degree, a random quantity with
mean `n_out K / n_in`, and a hidden unit has degree `K_l + d^+_v`. So
`K_G >= max_l K_l` with equality only in degenerate cases. `K` is clamped to
`[1, n_in]`, and `K >= n_in` produces a dense `eqx.nn.Linear`. The PDF states
this with equations.

## Usage

Everything lives on NFS (`/mnt/data0/shared/colin`); the environment is built on
the login node and jobs run on compute nodes through `uv run --no-sync`.

```bash
NFS=/mnt/data0/shared/colin
export PATH=$NFS/bin:$PATH UV_PYTHON_INSTALL_DIR=$NFS/.uv/python UV_CACHE_DIR=$NFS/.uv/cache
uv sync --python 3.12                     # jax[cuda12]==0.11.2, equinox 0.13.8, st and z1t editable

# CelebA train split -> data/celeba64_train.npy (uint8, 162770 x 64 x 64 x 3), about one minute
sbatch --job-name=celeba-prep slurm/cpu.sbatch python -m sparse_dsb.data \
  --zip /mnt/data0/shared/andrazj/jax_gibbs/misc_utils/data/downloads/manual/celeb_a/img_align_celeba.zip \
  --partition /mnt/data0/shared/dominic/diffusion_schrodinger_bridge/data/celeba/list_eval_partition.txt \
  --out data/celeba64_train.npy --image-size 64 --workers 16

sbatch --job-name=tests slurm/run.sbatch python -m pytest -x -q tests
sbatch --job-name=st-smoke slurm/run.sbatch python -m sparse_dsb.train --config configs/celeba64_smoke.yaml
sbatch --job-name=z1t-smoke slurm/run.sbatch python -m sparse_dsb.train --config configs/celeba64_z1t_smoke.yaml
```

A run writes to `runs/<name>/`: `config.yaml`, `info.json` (parameter counts,
`T`, prior mean), `logs/metrics.csv` (loss and gradient norm every `log_stride`
steps), `logs/plot_logs.csv`, `checkpoints/` and `im/`. Trailing `key=value`
arguments override config fields, e.g. `ipf.num_iter=500 model.n_layers=4`.

The block diagram is compiled with
`/mnt/data0/shared/colin/.tools/tectonic/tectonic report/unet_celeba.tex`.
