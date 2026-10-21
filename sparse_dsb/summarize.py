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


def montage(run_dir: str, path: str, directions=("b",), which="im_grid_final.png", max_columns: int = 6):
    """One row per direction, one column per (n, i) sample grid, cropped to the first two grid rows."""
    im_dir = os.path.join(run_dir, "im")
    tiles = []
    for direction in directions:
        names = sorted((d for d in os.listdir(im_dir) if d.startswith(direction + "_")),
                       key=lambda d: tuple(int(p) for p in d.split("_")[1:]))
        names = [d for d in names if os.path.exists(os.path.join(im_dir, d, which))]
        if len(names) > max_columns:
            names = names[-max_columns:]
        row = []
        for name in names:
            image = Image.open(os.path.join(im_dir, name, which))
            width, height = image.size
            crop = image.crop((0, 0, width, min(height, 2 * (64 + 2) + 2)))
            row.append((name, crop))
        tiles.append(row)
    if not tiles or not tiles[0]:
        return None
    tile_width = max(t.size[0] for row in tiles for _, t in row)
    tile_height = max(t.size[1] for row in tiles for _, t in row)
    label_height = 14
    columns = max(len(row) for row in tiles)
    sheet = Image.new("RGB", (columns * (tile_width + 4), len(tiles) * (tile_height + label_height + 4)), "white")
    from PIL import ImageDraw
    draw = ImageDraw.Draw(sheet)
    for r, row in enumerate(tiles):
        for c, (name, tile) in enumerate(row):
            x, y = c * (tile_width + 4), r * (tile_height + label_height + 4)
            draw.text((x + 2, y), name, fill="black")
            sheet.paste(tile, (x, y + label_height))
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
