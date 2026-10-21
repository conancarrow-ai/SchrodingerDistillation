"""Step-size schedule of the dsb reference code.

gammas has num_steps entries: a ramp of num_steps // 2 values from gamma_min to
gamma_max (linear or geometric) mirrored onto itself. time[k] is the cumulative
time after step k and T = time[-1] is the horizon. The networks are trained with
the input T - time[k] and sampled with the input time[k]; both conventions are
those of the reference code (see sparse_dsb/ipf.py).
"""
from __future__ import annotations

import numpy as np


def make_gammas(num_steps: int, gamma_min: float, gamma_max: float, gamma_space: str = "linspace") -> np.ndarray:
    half = num_steps // 2
    if gamma_space == "linspace":
        ramp = np.linspace(gamma_min, gamma_max, half)
    elif gamma_space == "geomspace":
        ramp = np.geomspace(gamma_min, gamma_max, half)
    else:
        raise ValueError(f"unknown gamma_space {gamma_space!r}")
    gammas = np.concatenate([ramp, np.flip(ramp)])
    if gammas.shape[0] != num_steps:
        raise ValueError("num_steps must be even")
    return gammas.astype(np.float32)


def cumulative_time(gammas: np.ndarray) -> np.ndarray:
    return np.cumsum(gammas).astype(np.float32)
