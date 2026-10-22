"""Summarise the toy-distribution runs: a table of energy distances and the final sample figures.

    python -m sparse_dsb.toy_summary [--runs runs] [--out report/toy]

Writes <out>/energy_distances.md (one row per run: the energy distance of the backward samples to the
data at the last and at the best IPF iteration, of the forward samples to the prior at the last
iteration and of the reference chain at n = 0) and copies the last backward and forward sample
figures of every run to <out>/<run>_b.png and <out>/<run>_f.png.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
from pathlib import Path

VARIANTS = ("dense", "maskfull", "mask32", "gather32")
DISTRIBUTIONS = ("moon", "scurve", "circle", "mixture", "mixturecentred")
FIGURE_DIRECTORY = re.compile(r"^([bf])_(\d+)_(\d+)$")


def latest_figure(run_dir: Path, direction: str) -> Path | None:
    im_dir = run_dir / "im"
    if not im_dir.is_dir():
        return None
    candidates = []
    for entry in im_dir.iterdir():
        match = FIGURE_DIRECTORY.match(entry.name)
        if match and match.group(1) == direction and (entry / "samples.png").exists():
            candidates.append((int(match.group(2)), int(match.group(3)), entry / "samples.png"))
    return max(candidates)[2] if candidates else None


def summarise_run(run_dir: Path) -> dict:
    rows = list(csv.DictReader(open(run_dir / "logs" / "toy_metrics.csv")))
    backward = [(int(row["n"]), float(row["energy_to_data"])) for row in rows if row["fb"] == "b"]
    forward = [(int(row["n"]), float(row["energy_to_prior"])) for row in rows if row["fb"] == "f"]
    info = json.load(open(run_dir / "info.json")) if (run_dir / "info.json").exists() else {}
    done = json.load(open(run_dir / "done.json")) if (run_dir / "done.json").exists() else {}
    return {
        "run": run_dir.name, "model": info.get("model", ""), "active": info.get("parameters_active"),
        "prior_mean": info.get("mean_final_vector"), "prior_var": info.get("var_final"),
        "last_n": backward[-1][0] if backward else None,
        "backward_last": backward[-1][1] if backward else None,
        "backward_best": min(backward, key=lambda item: item[1]) if backward else None,
        "forward_last": forward[-1][1] if forward else None,
        "reference_chain": next((value for n, value in forward if n == 0), None),
        "seconds": done.get("seconds"),
    }


def order_key(summary: dict) -> tuple:
    stem, variant = summary["run"].removeprefix("toy_").rsplit("_", 1)
    return (DISTRIBUTIONS.index(stem) if stem in DISTRIBUTIONS else 99, stem, VARIANTS.index(variant) if variant in VARIANTS else 99)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", default="runs")
    parser.add_argument("--out", default="report/toy")
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    run_dirs = sorted(path for path in Path(args.runs).glob("toy_*") if (path / "logs" / "toy_metrics.csv").exists())
    summaries = sorted((summarise_run(run_dir) for run_dir in run_dirs), key=order_key)
    lines = ["| run | model | active parameters | prior | last n | backward to data (last) | backward to data (best, n) | "
             "forward to prior (last) | reference chain to prior (n = 0) | seconds |", "|---|---|---|---|---|---|---|---|---|---|"]
    for summary in summaries:
        best = summary["backward_best"]
        prior = ""
        if summary["prior_mean"] is not None:
            prior = f"N(({', '.join(f'{v:.2f}' for v in summary['prior_mean'])}), {summary['prior_var']:.2f} I)"
        lines.append("| " + " | ".join([
            summary["run"], summary["model"], f"{summary['active']:,}" if summary["active"] else "", prior,
            str(summary["last_n"] or ""), f"{summary['backward_last']:.3f}" if summary["backward_last"] is not None else "",
            f"{best[1]:.3f} (n = {best[0]})" if best else "", f"{summary['forward_last']:.3f}" if summary["forward_last"] is not None else "",
            f"{summary['reference_chain']:.3f}" if summary["reference_chain"] is not None else "",
            f"{summary['seconds']:.0f}" if summary["seconds"] else "running"]) + " |")
    (out / "energy_distances.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for run_dir in run_dirs:
        for direction in ("b", "f"):
            source = latest_figure(run_dir, direction)
            if source is not None:
                shutil.copyfile(source, out / f"{run_dir.name}_{direction}.png")
    print("\n".join(lines))
    print(f"{len(summaries)} runs summarised in {out}")


if __name__ == "__main__":
    main()
