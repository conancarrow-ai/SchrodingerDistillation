"""Summarise a run directory: loss curves, per-half-step statistics and a montage of sample grids.

    python -m sparse_dsb.summarize runs/celeba64_st_smoke
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
from PIL import Image

from .plots import plot_losses


def half_step_table(metrics_csv: str) -> list[dict]:
    rows = np.genfromtxt(metrics_csv, delimiter=",", names=True, dtype=None, encoding="utf-8")
    table = []
    for n in np.unique(rows["n"]):
        for direction in ("b", "f"):
            mask = (rows["n"] == n) & (rows["fb"] == direction)
            if not mask.any():
                continue
            losses, iterations = rows["loss"][mask], rows["i"][mask]
            last = losses[iterations >= iterations.max() * 0.9]
            table.append({"n": int(n), "fb": direction, "first_loss": float(losses[0]), "last_10pct_loss": float(last.mean()),
                          "min_loss": float(losses.min()), "iterations": int(iterations.max()),
                          "seconds": float(rows["elapsed"][mask].max())})
    return table


def montage(run_dir: str, path: str, directions=("b",), which="im_grid_final.png", max_rows: int = 8, grid_rows: int = 1):
    """One labelled row per (direction, n, i) sample grid, each cropped to its first `grid_rows` rows of images."""
    im_dir = os.path.join(run_dir, "im")
    tiles = []
    for direction in directions:
        names = sorted((d for d in os.listdir(im_dir) if d.startswith(direction + "_")),
                       key=lambda d: tuple(int(p) for p in d.split("_")[1:]))
        names = [d for d in names if os.path.exists(os.path.join(im_dir, d, which))]
        for name in names[-max_rows:]:
            image = Image.open(os.path.join(im_dir, name, which))
            width, height = image.size
            tiles.append((name, image.crop((0, 0, width, min(height, grid_rows * 66 + 2)))))
    if not tiles:
        return None
    from PIL import ImageDraw
    tile_width = max(t.size[0] for _, t in tiles)
    label_height = 14
    sheet = Image.new("RGB", (tile_width, sum(t.size[1] + label_height + 2 for _, t in tiles)), "white")
    draw = ImageDraw.Draw(sheet)
    y = 0
    for name, tile in tiles:
        draw.text((2, y), name, fill="black")
        sheet.paste(tile, (0, y + label_height))
        y += tile.size[1] + label_height + 2
    sheet.save(path)
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    args = parser.parse_args()
    metrics = os.path.join(args.run_dir, "logs", "metrics.csv")
    table = half_step_table(metrics)
    with open(os.path.join(args.run_dir, "summary.json"), "w") as handle:
        json.dump(table, handle, indent=2)
    plot_losses(metrics, os.path.join(args.run_dir, "loss.png"))
    montage(args.run_dir, os.path.join(args.run_dir, "samples_backward.png"), directions=("b",))
    montage(args.run_dir, os.path.join(args.run_dir, "samples_forward.png"), directions=("f",))
    print(f"{'n':>3} {'fb':>2} {'first':>10} {'last10%':>10} {'min':>10} {'iters':>6} {'sec':>7}")
    for row in table:
        print(f"{row['n']:>3} {row['fb']:>2} {row['first_loss']:>10.5f} {row['last_10pct_loss']:>10.5f} "
              f"{row['min_loss']:>10.5f} {row['iterations']:>6d} {row['seconds']:>7.0f}")


if __name__ == "__main__":
    main()
