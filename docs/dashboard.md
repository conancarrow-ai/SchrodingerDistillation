# Monitoring dashboard

A static page, `dashboard/index.html` plus PNG figures, rewritten every two minutes
by `sparse_dsb/dashboard.py` on a compute node and served from fokker over the
tailnet.

URL: http://100.92.190.10:8892/ (equivalently http://fokker.tail02e69.ts.net:8892/).

## Start

```bash
cd /mnt/data0/shared/colin/Projects/SchrodingerDistillation
sbatch slurm/dashboard_loop.sbatch     # regenerator loop on a compute node (partition main, 2 CPUs, 6 GB, 7 days)
bash scripts/serve_dashboard.sh        # http.server on fokker, 100.92.190.10:8892, idempotent
```

The loop job logs one line per cycle to `slurm/logs/dashboard-loop-<jobid>.out`
(runs seen, figures redrawn, notes, seconds). The server writes
its log and pid to `.scratch/dashboard_state/server.{log,pid}`, outside the served
directory; stop it with `kill $(cat .scratch/dashboard_state/server.pid)`. The
loop job is named `sdsb-dashboard`. A single regeneration, for instance after
editing the script:

```bash
sbatch slurm/cpu.sbatch python -m sparse_dsb.dashboard --out dashboard --once
```

`dashboard/` is generated and ignored by git.

## Tabs

Runs: one row per directory under `runs/`: family, blocks, width, fan-in K,
patch size, parameters stored / active (`info.json`), IPF iterations x steps per
half-step (`config.yaml`), status, current half-step and step with the fraction
of the run completed, latest loss, mean loss over the last 10% of the logged
steps of the latest half-step, elapsed time and steps per second over the same
window (cache refreshes included). Status comes from `squeue` (a job is matched
to a run by the command echoed at the head of its log in `slurm/logs/`, else by
its name contained in the run name), from `done.json` and from the age of
`metrics.csv`: finished, running (with a warning when no metrics row arrived for
20 minutes), queued, running? (fresh metrics but no matched job), unknown (no
metrics and no job), failed (stale metrics and no job).
Below the table, one collapsible section per run (open for running runs):

- the loss curves of `sparse_dsb.plots.plot_losses` (every logged step, and one
  curve per half-step);
- chain statistics from `logs/plot_logs.csv`: final variance (log scale) and
  final mean of the 100 sampled chains at every checkpoint against the half-step
  index (b1, f1, b2, ...; ref is the reference chain before training), with the
  data variance (fixed 1,000-image sample of the array), the prior variance
  (`var_final`) and the shared mean (`mean_final` of `info.json`) as dashed
  lines; backward chains start from the prior and should reach the data
  statistics, forward chains start from data and should reach the prior;
- the latest backward and forward sample grids (`im_grid_final.png` of the
  latest `b_*` and `f_*` directories);
- the montages of `sparse_dsb.summarize.montage`: the first row of the final
  grid at the last eight checkpoints of each direction;
- the half-step table (first, last-10% and minimum loss, steps logged, seconds),
  the config and the tail of the job log.

Data: 100 training images drawn with seed 0 from the memory-mapped
`data/celeba64_train.npy`, the facts of `data/celeba64_train.json` and the pixel
mean, variance and channel means of a fixed 1,000-image sample.

Queue: this user's Slurm jobs (id, name, state, time, limit, partition, node or
pending reason, the run a job writes to) and the dashboard's own last
regeneration time, host and cycle duration.

Toy distribution validation: one matrix of the final backward samples of the toy runs (`runs/toy_*`,
rows the distributions with their priors, columns the networks: dense, mask K=full, mask K=32, gather
K=32) drawn from the saved coordinates `im/<fb>_<n>_<i>/samples.npz`, a strip of the energy distance to the
data along the IPF iterations, a table of the last and best distances, and the forward matrix in a
collapsed section. `python -m sparse_dsb.toy_resample` writes the coordinates for runs that predate them.

Benchmark: `report/benchmark_rtx6000.txt` verbatim.

## Regeneration cost

Figures are cached by a signature of their sources (size and mtime of
`metrics.csv`, `plot_logs.csv`, the sample grids, the data array) in
`.scratch/dashboard_state/cache.json` (`--state`) and redrawn only when a source changed; a cycle with
nothing new takes well under a second. A half-written CSV row is skipped, a
missing file or a failed figure shows as a note in the run's section, and the
page is written atomically. The dashboard only reads the files the training jobs
wrote.
