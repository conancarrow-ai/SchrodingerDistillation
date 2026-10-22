"""Write im/<fb>_<n>_<i>/samples.npz for the latest checkpoints of toy runs that predate it.

    python -m sparse_dsb.toy_resample [--runs runs] [--pattern 'toy_*'] [--all]

Rebuilds the run's IPF object from config.yaml, loads the latest net_<fb>_<n>_<i>.eqx of each direction
(every checkpoint with --all) and repeats the sampling of IPF.plot_samples (the same keys), so the saved
coordinates are those of the run's figures.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import equinox as eqx
import jax
import numpy as np
import yaml

from .ipf import IPF, IPFConfig, place
from .toy import ToyModelConfig, toy_points

CHECKPOINT = re.compile(r"^net_([bf])_(\d+)_(\d+)\.eqx$")


def checkpoints(run_dir: Path, every: bool) -> list[tuple[str, int, int, Path]]:
    found = []
    for path in (run_dir / "checkpoints").glob("net_*.eqx"):
        match = CHECKPOINT.match(path.name)
        if match:
            found.append((match.group(1), int(match.group(2)), int(match.group(3)), path))
    if every:
        return sorted(found)
    latest = {}
    for direction, n, i, path in found:
        if direction not in latest or (n, i) > latest[direction][1:3]:
            latest[direction] = (direction, n, i, path)
    return sorted(latest.values())


def resample(run_dir: Path, every: bool = False):
    config = yaml.safe_load(open(run_dir / "config.yaml"))
    data = config["data"]
    points = toy_points(data["distribution"], int(data.get("npar", 10000)), int(data.get("seed", 0)))
    ipf = IPF(IPFConfig(**config["ipf"]), ToyModelConfig(**config["model"]), points, str(run_dir))
    for direction, n, i, path in checkpoints(run_dir, every):
        target = run_dir / "im" / f"{direction}_{n}_{i}" / "samples.npz"
        if target.exists():
            print(f"{run_dir.name} {direction} n={n} i={i}: samples.npz present", flush=True)
            continue
        ipf.nets[direction] = place(eqx.tree_deserialise_leaves(path, ipf.nets[direction]), ipf.replicated)
        ipf.ema[direction] = eqx.filter(ipf.nets[direction], eqx.is_inexact_array)
        batch_key, chain_key = jax.random.split(jax.random.PRNGKey(0))
        batch = ipf._prior_batch(batch_key, ipf.config.plot_npar) if direction == "b" else ipf._data_batch(batch_key, ipf.config.plot_npar)
        params, static = eqx.partition(ipf.ema_net(direction), eqx.is_inexact_array)
        x_tot = ipf._sample(chain_key, params, static, batch)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez(target, init=np.asarray(batch, np.float32), final=np.asarray(x_tot[-1], np.float32))
        print(f"{run_dir.name} {direction} n={n} i={i}: wrote {target.relative_to(run_dir)}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", default="runs")
    parser.add_argument("--pattern", default="toy_*")
    parser.add_argument("--all", action="store_true", help="every checkpoint, not only the latest of each direction")
    args = parser.parse_args(argv)
    for run_dir in sorted(Path(args.runs).glob(args.pattern)):
        if (run_dir / "config.yaml").exists() and (run_dir / "checkpoints").is_dir():
            resample(run_dir, args.all)


if __name__ == "__main__":
    main()
