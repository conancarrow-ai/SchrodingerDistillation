"""Ring occupancy of the circle runs: the share of backward samples near each circle, per IPF iteration.

    python -m sparse_dsb.toy_rings [--runs runs] [--pattern 'toy_circle*']

For every im/b_<n>_<i>/samples.npz of the matching runs: the fraction of the final samples within one
unit of the outer circle (radius 10) and of the inner circle (radius 5); the data has half its points on
each. Prints one line per run with the per-iteration outer fractions and the means over the second half
of the iterations.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

SAMPLES = re.compile(r"^b_(\d+)_(\d+)$")


def ring_fractions(final: np.ndarray, width: float = 1.0) -> tuple[float, float]:
    radius = np.linalg.norm(final, axis=1)
    return float((np.abs(radius - 10.0) < width).mean()), float((np.abs(radius - 5.0) < width).mean())


def run_fractions(run_dir: Path) -> list[tuple[int, float, float]]:
    rows = []
    for entry in (run_dir / "im").iterdir():
        match = SAMPLES.match(entry.name)
        if match and (entry / "samples.npz").exists():
            outer, inner = ring_fractions(np.load(entry / "samples.npz")["final"])
            rows.append((int(match.group(1)), outer, inner))
    return sorted(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", default="runs")
    parser.add_argument("--pattern", default="toy_circle*")
    args = parser.parse_args(argv)
    for run_dir in sorted(Path(args.runs).glob(args.pattern)):
        rows = run_fractions(run_dir)
        if not rows:
            continue
        late = [row for row in rows if row[0] > max(row[0] for row in rows) // 2]
        outer_late, inner_late = np.mean([row[1] for row in late]), np.mean([row[2] for row in late])
        print(f"{run_dir.name:32s} outer {outer_late:.2f} inner {inner_late:.2f} (mean over n > {max(row[0] for row in rows) // 2}); "
              f"outer per n: {' '.join(f'{row[1]:.2f}' for row in rows)}")


if __name__ == "__main__":
    main()
