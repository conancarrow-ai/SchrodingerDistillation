"""Image grids and loss curves."""
from __future__ import annotations

import os

import numpy as np
from PIL import Image


def save_image_grid(images, path: str, nrow: int = 10, padding: int = 2):
    """images: (B, C, H, W) float in [0, 1]; writes a grid like torchvision.utils.save_image."""
    images = np.clip(np.asarray(images, dtype=np.float32), 0.0, 1.0)
    batch, channels, height, width = images.shape
    ncol = min(nrow, batch)
    nrows = (batch + ncol - 1) // ncol
    grid = np.ones((channels, nrows * (height + padding) + padding, ncol * (width + padding) + padding), np.float32)
    for index in range(batch):
        row, col = divmod(index, ncol)
        top, left = padding + row * (height + padding), padding + col * (width + padding)
        grid[:, top:top + height, left:left + width] = images[index]
    array = (grid * 255.0 + 0.5).astype(np.uint8).transpose(1, 2, 0)
    if channels == 1:
        array = array[..., 0]
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    Image.fromarray(array).save(path)


def plot_losses(metrics_csv: str, path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = np.genfromtxt(metrics_csv, delimiter=",", names=True, dtype=None, encoding="utf-8")
    if rows.size == 0:
        return
    figure, axes = plt.subplots(1, 2, figsize=(11, 3.5))
    for direction, color in (("b", "tab:blue"), ("f", "tab:orange")):
        mask = rows["fb"] == direction
        if not mask.any():
            continue
        iteration = rows["n"][mask]
        global_step = rows["global_step"][mask]
        axes[0].plot(global_step, rows["loss"][mask], ".", ms=2, color=color, label=f"{direction} half-steps")
        for n in np.unique(iteration):
            select = iteration == n
            axes[1].plot(rows["i"][mask][select], rows["loss"][mask][select], color=color, alpha=0.3 + 0.7 * n / iteration.max())
    axes[0].set_yscale("log"); axes[0].set_xlabel("global step"); axes[0].set_ylabel("mean-matching loss"); axes[0].legend()
    axes[1].set_yscale("log"); axes[1].set_xlabel("iteration within a half-step"); axes[1].set_title("one curve per half-step (darker = later IPF iteration)")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
