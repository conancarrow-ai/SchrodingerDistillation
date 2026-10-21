# Data-parallel training over the GPUs of one node

`sparse_dsb/ipf.py` trains on every GPU a job can see. The GPUs form a
one-dimensional mesh with axis `batch`; each gradient step evaluates and
differentiates the network on its share of the minibatch, XLA sums the
gradients, and every device applies the same Adam update to its replica of
the parameters. The cache of trajectories is built the same way: each device
simulates its share of the chains.

## Usage

```bash
# every GPU of the allocation (ipf.devices defaults to 0)
sbatch --job-name=st-4gpu --gres=gpu:rtx6000:4 slurm/run.sbatch \
  python -m sparse_dsb.train --config configs/celeba64_smoke.yaml --out runs/st_4gpu
# the first two of them, or an explicit list of indices into jax.devices()
sbatch --gres=gpu:rtx6000:4 slurm/run.sbatch python -m sparse_dsb.train --config ... ipf.devices=2
sbatch --gres=gpu:rtx6000:4 slurm/run.sbatch python -m sparse_dsb.train --config ... "ipf.devices=[1, 3]"
```

`IPFConfig.devices` (yaml key `ipf.devices`): `0` uses every visible device,
`n` the first `n`, a list those indices of `jax.devices()`. A job sees the GPUs
Slurm allocated through `CUDA_VISIBLE_DEVICES`, so `slurm/run.sbatch` with
`--gres=gpu:rtx6000:N` needs nothing else. `cache_npar`, `plot_npar` and
`batch_size` must be divisible by the number of devices; `IPF.__init__` raises
a `ValueError` naming the field otherwise. The defaults 300, 100 and 128 work
for 1, 2 and 4 devices only; 8 devices need, for instance, `cache_npar=304` and
`plot_npar=96` (3, 5, 6 and 7 devices fail on `batch_size` or `plot_npar` as well). `info.json`
records the devices of a run.

## What is sharded

| object | sharding | where it is set |
|---|---|---|
| parameters, optimiser state, EMA parameters, gradients | replicated, `PartitionSpec()` | `place` on construction, checkpoint load and `optimizer.init`; `eqx.filter_shard` at the entry and exit of the jitted train step and on the gradients |
| data array (uint8 images) | replicated | `place` in `IPF.__init__` |
| data batch, prior batch | rows, `PartitionSpec("batch")` | `with_sharding_constraint` on the output of `data_batch` and `prior_batch` |
| chains (`record_init_langevin`, `record_langevin_seq`) | rows of the batch axis | the initial state is constrained before the `lax.scan`, so the carry and every network evaluation inside the scan are sharded |
| trajectory cache `(xs, outs, steps)` | rows | constraint on the output of `cache_init` / `cache_from_net`, and `place` after `concatenate_caches` |
| minibatch of a step | rows | constraint after `minibatch` inside the train step |
| sample chains (`sample`, `init_trajectories`) | rows of the batch axis | constraint on the initial state |

The minibatch gather stays inside the jitted step. GSPMD partitions the gather
of `batch_size` random rows from the row-sharded cache as a local masked
gather on every device followed by an all-reduce of the minibatch (a few MB);
the compiled step contains no all-gather of the cache. The only other
collective is the all-reduce of the gradient tuple.

Random numbers: `jax_threefry_partitionable` (the default since jax 0.5) makes
`jax.random.normal` produce the same values whatever the sharding of its
output, so the chain noise, the prior samples, the data indices and the
horizontal flips of a sharded run are those of the single-device run with
the same seed. Only the order of float32 reductions (the loss mean and the
gradient sums) differs.

Checkpoints: `eqx.tree_serialise_leaves` writes `np.asarray(leaf)`, which is
one copy of a replicated array, so the `.eqx` files of a sharded run are byte
for byte those a single-device run would write. Loading deserialises on the
default device and `place` puts the leaves on the replicated sharding.

With one device the mesh has one entry and every constraint is a no-op, so
the program is the previous single-device program (verified bit for bit, see
below). The public API (`IPF.__init__`, `ipf_step`, `train`, `new_cache`,
`record_*`, `flatten_trajectories`, `minibatch`, `ModelConfig`, `build_model`,
`parameter_counts`) is unchanged; `IPF` gained the attributes `devices`,
`mesh`, `replicated` and `batch_sharding` and the method `new_permutation`.

## Verification

All runs on RTX PRO 6000 Blackwell (mklgpusrv3, mklgpusrv2) through
`slurm/run.sbatch`; job logs are in `slurm/logs/<name>-<jobid>.out`.

### Sharded against single-device (`tests/test_sharding.py`, jobs 694089 and 694131, two GPUs)

For each of st/mask, z1t/mask and z1t/gather the tiny synthetic IPF smoke of
`tests/test_sparse_dsb.py` (8x8 images, 6 steps, 2 IPF iterations of 12 steps,
batch 16, cache of 2 x 8 trajectories, 8 plot samples) ran on both GPUs
(`devices=0`) and on GPU 0 (`devices=[0]`) with the same seed.

| family | max relative difference of the 16 logged losses | of the gradient norms | sample statistics (init/final var and mean, 9 grids) | grid pixels differing |
|---|---|---|---|---|
| st/mask | 1.7e-7 | 7.5e-6 | 0 | 0 of 18 grids |
| z1t/mask | 1.3e-7 | 1.1e-7 | 0 | 0 |
| z1t/gather | 1.4e-7 | 1.2e-7 | 0 | 0 |

After the review the test uses an EMA rate of 0.5 so that the sampled grids
reflect the trained networks, and compares the trained parameters of the two
runs leaf by leaf (job 694341: max absolute difference 1.1e-6 for st, 1.2e-7
for z1t, with parameter magnitudes up to 3; losses within 1.9e-7; grids
identical). The test records these numbers as pytest properties, which the suite's
conftest prints in the terminal summary (`python -m pytest -q tests/test_sharding.py`).

The test also checks that the masks (st) and gather indices (z1t) of both
nets are identical between the two runs and identical on every device
(replicated arrays, one shard per device); that `net_b_2_12.eqx` of the
sharded run is byte for byte the file written from a host copy of the same
net (single copy); that the four checkpoints of the sharded run load into a
single-device `IPF` (`checkpoint_run=True`, iteration 2, pass f) with leaves
equal to the sharded run's, EMA outputs on a fixed batch equal within 1e-5,
the backward sample grid equal, and that training resumes from them with
finite losses. It skips when fewer than two GPUs are visible. The tolerance
is 1e-3; the differences come from the order of float32 reductions (the
2-device run sums two partial means and two partial gradients).

### Single device against the previous code (job 694089)

`.scratch/sharding/compare_old_new.py` ran the same smoke with the previous
`ipf.py` (no mesh) and the new one on one GPU: the logged losses and gradient
norms, all checkpoints and the sample grids are bit-identical for st and z1t.

### Existing tests (job 694068, one GPU)

`python -m pytest -x -q tests`: 12 passed, 4 skipped (the sharding tests).

### Throughput (jobs 694082 to 694087, `runs/scaling_<family>_<n>gpu/`)

`configs/celeba64_smoke.yaml` and `configs/celeba64_z1t_smoke.yaml` with
`ipf.num_iter=300 ipf.n_ipf=1 ipf.gif_stride=300`: batch 128, cache of 300
trajectories of 50 steps refreshed every 100 steps, one b and one f
half-step, on 1, 2 and 4 RTX PRO 6000 of mklgpusrv3 (the 1-GPU runs on
mklgpusrv2), one run at a time. Steps per second is the median over the
10-step logging windows above iteration 100 that contain no cache refresh
(`logs/metrics.csv`, `elapsed` column; `.scratch/sharding/throughput.py`);
the cache build time is the one printed in the job log for the refreshes
after the first (the first includes compilation). The b half-step of
iteration 1 builds its cache with the reference OU chain (no network, 0.0 s
at every device count), the f half-step with the EMA of the backward
network (two network evaluations per step for 300 x 50 steps).

| family | GPUs | steps/s, b half-step | steps/s, f half-step | f cache build (s) | loss at i = 10, b | loss at i = 10, f | wall time of the run (s) |
|---|---|---|---|---|---|---|---|
| st | 1 | 13.5 | 13.2 | 6.2 | 0.08345767855644226 | 0.09991499036550522 | 341 |
| st | 2 | 24.4 | 23.8 | 3.0 | 0.08345767855644226 | 0.09991499036550522 | 339 |
| st | 4 | 35.7 | 35.7 | 1.3 | 0.08345767110586166 | 0.09991499036550522 | 266 |
| z1t | 1 | 21.3 | 21.3 | 3.9 | 0.08363547921180725 | 0.10037227720022202 | 256 |
| z1t | 2 | 41.7 | 34.5 | 1.8 | 0.08363547921180725 | 0.10037227720022202 | 221 |
| z1t | 4 | 58.8 | 58.8 | 0.8 | 0.08363547921180725 | 0.10037227720022202 | 215 |

The 1-GPU runs were on mklgpusrv2 while other jobs shared that node; the 2- and
4-GPU runs on mklgpusrv3, so the ratios carry some node-load noise.
Speed-ups over one GPU: st 1.8x (2) and 2.6x (4) for the gradient steps,
2.1x and 4.8x for the cache build; z1t 1.6 to 2.0x (2) and 2.8x (4), cache
2.2x and 4.9x. The gradient step is 128 images of 513 tokens; at 32 images
per device the per-step work no longer fills a GPU, hence the sub-linear
scaling of the steps, while the cache build (300 chains in flight) scales
linearly. The runs agree: the first logged losses are equal to 7 digits
(st on 4 GPUs: relative difference 9e-8), the sample statistics at
i = 300 are identical to the four printed digits at every device count
(st: b final var 4.3371, mean 0.4012, f 4.2213, 0.4019; z1t: 4.5344,
0.4066, 4.3621, 0.4191), and the last-10% mean losses of both half-steps
are identical to the five printed digits (st 0.07511 / 0.09692, z1t
0.07634 / 0.09760). Wall times include 100 to 130 s of compilation, which
does not grow with the device count. The checkpoints of these runs were
deleted; logs, grids and job logs remain.

Memory: the uint8 image array (2 GB for CelebA 64) is replicated on every
GPU; the cache (0.7 GB at these settings) is split.
