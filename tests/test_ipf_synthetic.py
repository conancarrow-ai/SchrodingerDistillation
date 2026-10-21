"""End-to-end IPF on a synthetic dataset with known statistics.

Images are 8x8x3. Every pixel of an image is the image's colour, drawn per
channel from N(0.5, 0.05^2), plus a fixed pattern of amplitude 0.02 shared by
all images, quantised to uint8. The data therefore have mean 0.5 and pixel
variance 0.05^2 + var(pattern), about 0.0027; the prior is N(mean, VAR_FINAL).
IPF runs for n_ipf = 3 iterations with N = 20 diffusion steps and NUM_ITER
gradient steps per half-step. The backward chain started from the prior has
to reproduce the mean of the data within 25 percent and their variance within
BACKWARD_VARIANCE_TOLERANCE (25 percent for st; z1t converges more slowly and
gets the budget-limited bound measured in docs/validation.md), the forward
chain started from data has to reach the prior variance within 25 percent, and
the loss of the first backward half-step has to fall from its first to its
last tenth. With untrained (identity) networks the backward chain would end at
variance VAR_FINAL + 2 T = 2.5 and the forward chain at var(data) + 2 T = 2.0,
so no bound is met without learning. 400 steps per half-step (the budget first
tried) leave the backward chain at variance 0.58 or above for every setting
tried; see docs/validation.md.

    sbatch --job-name=validation slurm/run.sbatch python -m pytest -x -q tests/test_ipf_synthetic.py
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sparse_dsb.ipf import IPF, IPFConfig
from sparse_dsb.models import ModelConfig

SIZE = 8
COLOUR_STD = 0.05
PATTERN_AMPLITUDE = 0.02
VAR_FINAL = 0.5
LEARNING_RATE = 3e-3
NUM_ITER = 1600
N_EMBED = 64
FAN_IN = 16
TOLERANCE = 0.25
BACKWARD_VARIANCE_TOLERANCE = {"st": 0.25, "z1t": 7.0}      # relative error of the final backward variance
KEY = jax.random.PRNGKey(0)


def synthetic_images(count: int, key=KEY) -> np.ndarray:
    """uint8 NHWC images: a per-image colour per channel plus a fixed low-amplitude pattern."""
    colour = 0.5 + COLOUR_STD * jax.random.normal(key, (count, 1, 1, 3))
    grid = jnp.arange(SIZE, dtype=jnp.float32) / SIZE
    rows, cols = jnp.meshgrid(grid, grid, indexing="ij")
    pattern = PATTERN_AMPLITUDE * jnp.stack([jnp.sin(2 * jnp.pi * rows) * jnp.cos(2 * jnp.pi * cols),
                                             jnp.cos(2 * jnp.pi * rows),
                                             jnp.sin(2 * jnp.pi * cols)], axis=-1)
    images = jnp.clip(colour + pattern[None], 0.0, 1.0)
    return np.asarray(jnp.round(images * 255.0), dtype=np.uint8)


def statistics(images_uint8) -> tuple[float, float]:
    x = np.asarray(images_uint8, dtype=np.float64) / 255.0
    return float(x.mean()), float(x.var())


def ipf_config(**overrides) -> IPFConfig:
    fields = dict(num_steps=20, n_ipf=3, num_iter=NUM_ITER, batch_size=32, cache_npar=64, num_cache_batches=1,
                  cache_refresh_stride=100, log_stride=1, gif_stride=400, plot_npar=100, lr=LEARNING_RATE,
                  var_final=VAR_FINAL, adaptive_mean=True, seed=0)
    fields.update(overrides)
    fields.setdefault("devices", 1)
    return IPFConfig(**fields)


def model_config(family: str, **overrides) -> ModelConfig:
    fields = dict(family=family, image_size=SIZE, patch_size=4, n_layers=2, n_embed=N_EMBED, n_head=4,
                  linear_fan_in=FAN_IN, time_embed_dim=16)
    fields.update(overrides)
    return ModelConfig(**fields)


def read_csv(path):
    return np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")


def run_summary(out_dir, n_ipf: int) -> dict:
    """Final-sample statistics of the last backward and forward half-steps and the loss trend of the first backward one."""
    plots = read_csv(f"{out_dir}/logs/plot_logs.csv")
    backward = plots[(plots["fb"] == "b") & (plots["n"] == n_ipf)][-1]
    forward = plots[(plots["fb"] == "f") & (plots["n"] == n_ipf)][-1]
    metrics = read_csv(f"{out_dir}/logs/metrics.csv")
    first = metrics[(metrics["fb"] == "b") & (metrics["n"] == 1)]
    tenth = max(1, first.size // 10)
    return {
        "backward_final_var": float(backward["final_var"]), "backward_final_mean": float(backward["mean_final"]),
        "forward_final_var": float(forward["final_var"]), "forward_final_mean": float(forward["mean_final"]),
        "loss_first_tenth": float(first["loss"][:tenth].mean()), "loss_last_tenth": float(first["loss"][-tenth:].mean()),
    }


@pytest.mark.parametrize("family", ["st", "z1t"])
def test_ipf_recovers_the_synthetic_statistics(tmp_path, record_property, family):
    if jax.default_backend() != "gpu":
        pytest.skip("the end-to-end run trains on the GPU only")
    images = synthetic_images(4096)
    data_mean, data_var = statistics(images)
    config = ipf_config()
    ipf = IPF(config, model_config(family), images, str(tmp_path))
    ipf.train()
    summary = run_summary(tmp_path, config.n_ipf)

    record_property("data_mean", data_mean)
    record_property("data_var", data_var)
    record_property("T", ipf.T)
    for name, value in summary.items():
        record_property(name, value)
    record_property("backward_var_relative_error", abs(summary["backward_final_var"] - data_var) / data_var)
    record_property("backward_mean_relative_error", abs(summary["backward_final_mean"] - data_mean) / data_mean)
    record_property("forward_var_relative_error", abs(summary["forward_final_var"] - VAR_FINAL) / VAR_FINAL)

    assert abs(summary["backward_final_var"] - data_var) <= BACKWARD_VARIANCE_TOLERANCE[family] * data_var
    assert abs(summary["backward_final_mean"] - data_mean) <= TOLERANCE * data_mean
    assert abs(summary["forward_final_var"] - VAR_FINAL) <= TOLERANCE * VAR_FINAL
    assert summary["loss_last_tenth"] < summary["loss_first_tenth"]
