"""Static monitoring dashboard of the runs: <out>/index.html plus PNG figures, in four tabs.

Runs: one row per directory under runs/ (model, parameter counts, IPF schedule, status
from squeue, done.json and the age of the last metrics row, current half-step and step,
latest loss, mean loss over the last 10% of the latest half-step, elapsed time, steps per
second), then one collapsible section per run: the loss curves of
sparse_dsb.plots.plot_losses, the final variance and mean of the sampled chains against
the half-step index with the data and prior targets, the latest backward and forward
sample grids, the montages over half-steps of sparse_dsb.summarize.montage, the half-step
table, the config and the tail of the job log.
Data: 100 training images drawn with a fixed seed from the memory-mapped array, the
dataset facts and the pixel statistics of a 1,000-image sample.
Queue: this user's Slurm jobs and the dashboard's own regeneration time.
Benchmark: report/benchmark_rtx6000.txt verbatim.

A figure is redrawn only when the size or mtime of its sources changed, so a cycle with
nothing new takes well under a second. A half-written CSV or a missing file produces a
note in the page, never a crash. The dashboard reads what the training jobs wrote and
computes nothing on an accelerator.

Once: python -m sparse_dsb.dashboard --out dashboard --once
Loop: python -m sparse_dsb.dashboard --out dashboard --loop 120
Serve on fokker: scripts/serve_dashboard.sh
"""
from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import os
import re
import shutil
import socket
import subprocess
import time
import traceback
from pathlib import Path

import numpy as np
from PIL import Image
import yaml

from .plots import plot_losses, save_image_grid
from .summarize import montage

PROJECT = Path(__file__).resolve().parents[1]
RUNS = PROJECT / "runs"
DATA_ARRAY = PROJECT / "data" / "celeba64_train.npy"
DATA_FACTS = PROJECT / "data" / "celeba64_train.json"
BENCHMARK = PROJECT / "report" / "benchmark_rtx6000.txt"
SLURM_LOGS = PROJECT / "slurm" / "logs"
SCRIPT_STAMP = str(Path(__file__).resolve().stat().st_mtime_ns)
TABS = (("runs", "Runs"), ("toy", "Toy distribution validation"), ("data", "Data"), ("queue", "Queue"), ("benchmark", "Benchmark"))
TOY_DISTRIBUTIONS = ("moon", "scurve", "circle", "mixture")
TOY_VARIANTS = ("dense", "maskfull", "mask32", "gather32")
STALE_SECONDS = 20 * 60
ACTIVE_STATES = ("RUNNING", "COMPLETING", "CONFIGURING")
DIRECTION_LABEL = {"b": "backward (prior to data)", "f": "forward (data to prior)"}
RUN_LINE = re.compile(r"^=== run: (.*) ===$", re.M)
GRID_DIRECTORY = re.compile(r"^[bf]_\d+_\d+$")
LOG_RUN_DIRS: dict[Path, Path] = {}


# ---------------------------------------------------------------- small helpers
def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def stamp(path: Path, date_format: str = "%m-%d %H:%M UTC") -> str:
    return dt.datetime.fromtimestamp(path.stat().st_mtime, dt.timezone.utc).strftime(date_format)


def escape_html(text) -> str:
    return html.escape(str(text))


def duration(seconds) -> str:
    if seconds is None:
        return ""
    seconds = int(seconds)
    if seconds < 3600:
        return f"{seconds // 60} min"
    return f"{seconds // 3600} h {seconds % 3600 // 60:02d} min"


def file_signature(path: Path) -> str:
    try:
        status = path.stat()
        return f"{path.name}:{status.st_size}:{status.st_mtime_ns}"
    except OSError:
        return f"{path.name}:missing"


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def read_yaml(path: Path) -> dict:
    try:
        with open(path, encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}
    except (OSError, yaml.YAMLError):
        return {}


def read_table(path: Path) -> dict[str, np.ndarray]:
    """Columns of a CSV whose first column is a string; a trailing half-written row is skipped."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return {}
    if not lines:
        return {}
    header = lines[0].split(",")
    rows = []
    for line in lines[1:]:
        parts = line.split(",")
        if len(parts) != len(header):
            continue
        try:
            rows.append((parts[0], *[float(part) for part in parts[1:]]))
        except ValueError:
            continue
    return {name: np.array([row[column] for row in rows]) for column, name in enumerate(header)}


def tail_lines(path: Path, count: int = 6, size: int = 4096) -> list[str]:
    try:
        with open(path, "rb") as handle:
            handle.seek(max(0, os.path.getsize(path) - size))
            text = handle.read().decode(errors="replace")
    except OSError:
        return []
    return [line for line in text.splitlines() if line.strip()][-count:]


def write_atomic(path: Path, text: str):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def render(path: Path, draw):
    """draw(temporary_path), then an atomic rename, so a reader never sees a half-written PNG."""
    temporary = path.with_name(path.stem + ".tmp" + path.suffix)
    draw(str(temporary))
    if temporary.exists():
        os.replace(temporary, path)


class FigureCache:
    """Signatures of the sources behind each figure, kept in <state>/cache.json across cycles."""

    def __init__(self, state: Path):
        state.mkdir(parents=True, exist_ok=True)
        self.path = state / "cache.json"
        self.entries = read_json(self.path)
        self.redrawn = 0

    def ensure(self, name: str, signature: str, output: Path, draw) -> str | None:
        """Call draw() when the output is missing or its sources changed; return a note on failure."""
        if output.exists() and self.entries.get(name) == f"{SCRIPT_STAMP} {signature}":
            return None
        try:
            draw()
        except Exception as error:
            return f"{name}: {type(error).__name__}: {error}"
        if not output.exists():
            return f"{name}: nothing to draw yet"
        self.entries[name] = f"{SCRIPT_STAMP} {signature}"
        self.redrawn += 1
        return None

    def save(self):
        write_atomic(self.path, json.dumps(self.entries, indent=1))


# ---------------------------------------------------------------- Slurm
def slurm_jobs() -> tuple[list[dict], str]:
    command = ["squeue", "-u", str(os.getuid()), "-h", "-o", "%i|%j|%T|%M|%l|%P|%N|%R"]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired) as error:
        return [], f"squeue failed: {error}"
    if completed.returncode != 0:
        return [], f"squeue failed: {completed.stderr.strip()}"
    jobs = []
    for line in completed.stdout.splitlines():
        fields = line.split("|")
        if len(fields) != 8:
            continue
        job = dict(zip(("id", "name", "state", "time", "limit", "partition", "nodes", "reason"), fields))
        job["log"] = SLURM_LOGS / f"{job['name']}-{job['id']}.out"
        jobs.append(job)
    return jobs, ""


def run_dir_of_log(log: Path) -> Path | None:
    """The run directory a training job writes to, read from the command echoed at the head of its log."""
    if log in LOG_RUN_DIRS:
        return LOG_RUN_DIRS[log]
    try:
        with open(log, encoding="utf-8", errors="replace") as handle:
            head = handle.read(4096)
    except OSError:
        return None
    match = RUN_LINE.search(head)
    if not match or "sparse_dsb.train" not in match.group(1):
        return None
    tokens = match.group(1).split()

    def option(flag):
        return tokens[tokens.index(flag) + 1] if flag in tokens[:-1] else None

    run_dir = None
    if option("--out"):
        run_dir = (PROJECT / option("--out")).resolve()
    elif option("--config"):
        config = option("--config")
        run_dir = (RUNS / (read_yaml(PROJECT / config).get("name") or Path(config).stem)).resolve()
    if run_dir is not None:
        LOG_RUN_DIRS[log] = run_dir
    return run_dir


def job_matches_run(job_name: str, run_name: str) -> bool:
    """Fallback for jobs without a log yet (pending): the job name must be contained in the run name."""
    job, run = (re.sub(r"[^a-z0-9]", "", text.lower()) for text in (job_name, run_name))
    return len(job) >= 4 and job in run


# ---------------------------------------------------------------- runs
def half_step_rows(metrics: dict) -> list[dict]:
    rows = []
    if not metrics or not len(metrics["loss"]):
        return rows
    for n in np.unique(metrics["n"]):
        for direction in ("b", "f"):
            mask = (metrics["n"] == n) & (metrics["fb"] == direction)
            if not mask.any():
                continue
            losses, iterations = metrics["loss"][mask], metrics["i"][mask]
            last = losses[iterations >= 0.9 * iterations.max()]
            rows.append({"n": int(n), "fb": direction, "first_loss": float(losses[0]), "last_10pct_loss": float(last.mean()),
                         "min_loss": float(losses.min()), "iterations": int(iterations.max()),
                         "seconds": float(metrics["elapsed"][mask].max())})
    return rows


def run_status(run: dict, jobs_of_run: list[dict], metrics_path: Path, squeue_error: str) -> tuple[str, str, str]:
    running = [job for job in jobs_of_run if job["state"] in ACTIVE_STATES]
    pending = [job for job in jobs_of_run if job["state"] == "PENDING"]
    age = time.time() - metrics_path.stat().st_mtime if metrics_path.exists() else None
    if run["done"]:
        return "finished", f"done.json {stamp(run['dir'] / 'done.json')}", "ok"
    if running:
        job = running[0]
        detail = f"job {job['id']} on {job['nodes']}, {job['time']} of {job['limit']}"
        if age is not None and age > STALE_SECONDS:
            return "running", f"{detail}; no metrics row for {duration(age)}", "warn"
        return "running", detail, "ok"
    if pending:
        job = pending[0]
        return "queued", f"job {job['id']} {job['reason']}", "muted"
    if age is not None and age < STALE_SECONDS:
        reason = "squeue unavailable" if squeue_error else "no job matched"
        return "running?", f"{reason}; last metrics row {duration(age)} ago", "warn"
    if age is None:
        return "unknown", "no metrics.csv yet and no job matched", "muted"
    last = run["log_tail"][-1] if run["log_tail"] else ""
    return "failed", f"no job; last metrics row {stamp(metrics_path)}; {last[:90]}", "bad"


def scan_run(run_dir: Path, jobs: list[dict], log_runs: dict[Path, Path], squeue_error: str) -> dict:
    run = {"name": run_dir.name, "dir": run_dir, "notes": [], "config": {}, "info": {}, "done": {}, "model": {}, "ipf": {},
           "kind": "images", "distribution": None, "variant": "", "toy_metrics": {},
           "num_iter": 0, "n_ipf": 0, "metrics": {}, "plot_logs": {}, "metrics_signature": "", "latest": None,
           "elapsed": None, "half_steps": [], "log_tail": [], "latest_log": None,
           "status": "unknown", "status_detail": "", "status_class": "muted"}
    try:
        run["config"] = config = read_yaml(run_dir / "config.yaml")
        run["info"], run["done"] = read_json(run_dir / "info.json"), read_json(run_dir / "done.json")
        run["model"], run["ipf"] = config.get("model") or {}, config.get("ipf") or {}
        data = config.get("data") or {}
        run["kind"], run["distribution"] = data.get("kind", "images"), data.get("distribution")
        run["variant"] = run["name"].rsplit("_", 1)[-1] if run["kind"] == "toy" else ""
        run["toy_metrics"] = read_table(run_dir / "logs" / "toy_metrics.csv") if run["kind"] == "toy" else {}
        run["num_iter"], run["n_ipf"] = int(run["ipf"].get("num_iter") or 0), int(run["ipf"].get("n_ipf") or 0)
        if not config:
            run["notes"].append("config.yaml missing or unreadable")
        metrics_path = run_dir / "logs" / "metrics.csv"
        run["metrics"] = metrics = read_table(metrics_path)
        run["metrics_signature"] = file_signature(metrics_path)
        run["plot_logs"] = read_table(run_dir / "logs" / "plot_logs.csv")
        if metrics and len(metrics["loss"]):
            direction, n, i = str(metrics["fb"][-1]), int(metrics["n"][-1]), int(metrics["i"][-1])
            mask = (metrics["fb"] == direction) & (metrics["n"] == n)
            iterations, losses, elapsed = metrics["i"][mask], metrics["loss"][mask], metrics["elapsed"][mask]
            window = iterations >= 0.9 * iterations.max()
            if window.sum() < 2:
                window = np.ones_like(window)
            rate = None
            if elapsed[window][-1] > elapsed[window][0]:
                rate = float((iterations[window][-1] - iterations[window][0]) / (elapsed[window][-1] - elapsed[window][0]))
            progress = None
            if run["num_iter"] and run["n_ipf"]:
                progress = (2 * (n - 1) + (direction == "f") + i / run["num_iter"]) / (2 * run["n_ipf"])
            run["latest"] = {"fb": direction, "n": n, "i": i, "loss": float(losses[-1]),
                             "last_10pct_loss": float(losses[window].mean()), "rate": rate, "progress": progress}
        run["half_steps"] = half_step_rows(metrics)
        jobs_of_run = [job for job in jobs if log_runs.get(job["log"]) == run_dir or job_matches_run(job["name"], run_dir.name)]
        logs_of_run = sorted((log for log, directory in log_runs.items() if directory == run_dir), key=lambda log: log.stat().st_mtime)
        run["jobs"] = jobs_of_run
        run["latest_log"] = logs_of_run[-1] if logs_of_run else None
        run["log_tail"] = tail_lines(run["latest_log"]) if run["latest_log"] else []
        run["status"], run["status_detail"], run["status_class"] = run_status(run, jobs_of_run, metrics_path, squeue_error)
        if run["done"]:
            run["elapsed"] = run["done"].get("seconds")
        elif metrics_path.exists() and (run_dir / "config.yaml").exists():
            run["elapsed"] = metrics_path.stat().st_mtime - (run_dir / "config.yaml").stat().st_mtime
    except Exception as error:
        run["notes"].append(f"scan: {type(error).__name__}: {error}")
    return run


def grid_files(run_dir: Path, direction: str, filename: str = "im_grid_final.png") -> list[Path]:
    """im/<direction>_<n>_<i>/<filename> in half-step order."""
    im_dir = run_dir / "im"
    if not im_dir.is_dir():
        return []
    names = sorted((entry.name for entry in im_dir.iterdir() if GRID_DIRECTORY.match(entry.name) and entry.name[0] == direction),
                   key=lambda name: tuple(int(part) for part in name.split("_")[1:]))
    return [im_dir / name / filename for name in names if (im_dir / name / filename).exists()]


def model_label(run: dict) -> str:
    model = run["model"]
    if run["kind"] == "toy":
        kind = model.get("sparse_linear", "dense")
        return "dense" if kind == "dense" else f"{kind} K={model.get('linear_fan_in') or 'full'}"
    return str(model.get("family", ""))


def draw_chain_statistics(plot_logs: dict, num_iter: int, n_ipf: int, data_variance, data_mean, mean_final, var_final, path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(11, 3.5))
    n_max = max(1, n_ipf, int(plot_logs["n"].max()))
    for direction, color, marker in (("b", "tab:blue", "o"), ("f", "tab:orange", "s")):
        mask = plot_logs["fb"] == direction
        if not mask.any():
            continue
        x = (plot_logs["n"][mask] - 1) + (direction == "f") + plot_logs["i"][mask] / max(1, num_iter)
        order = np.argsort(x)
        for axis, column in zip(axes, ("final_var", "mean_final")):
            axis.plot(x[order], plot_logs[column][mask][order], marker=marker, ms=4, lw=1.5, color=color, label=DIRECTION_LABEL[direction])
    if data_variance is not None:
        axes[0].axhline(data_variance, ls="--", lw=1, color="tab:blue", alpha=0.7, label=f"data variance {data_variance:.3f}")
    if var_final is not None:
        axes[0].axhline(var_final, ls="--", lw=1, color="tab:orange", alpha=0.7, label=f"prior variance {var_final:.3f}")
    if data_mean is not None:
        axes[1].axhline(data_mean, ls="--", lw=1, color="tab:blue", alpha=0.7, label=f"data mean {data_mean:.3f}")
    if mean_final is not None:
        axes[1].axhline(mean_final, ls="--", lw=1, color="tab:orange", alpha=0.7, label=f"prior mean {mean_final:.3f} (mean of the first 100 images)")
    ticks = list(range(0, 2 * n_max + 1))
    labels = ["ref"] + [f"{direction}{n}" for n in range(1, n_max + 1) for direction in "bf"]
    for axis in axes:
        axis.set_xticks(ticks)
        axis.set_xticklabels(labels)
        axis.set_xlim(-0.15, 2 * n_max + 0.15)
        axis.set_xlabel("half-step; the position within one is the fraction of its steps")
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("final variance of the sampled chains")
    axes[1].set_ylabel("final mean of the sampled chains")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def run_figures(run: dict, out: Path, cache: FigureCache, data_variance, data_mean) -> dict:
    """Draw or refresh the figures of one run; returns {kind: path or None} and appends notes to the run."""
    name, run_dir, figures = run["name"], run["dir"], {}

    def note(text):
        if text:
            run["notes"].append(text)

    metrics = run["metrics"]
    loss_path = out / f"{name}_loss.png"

    def draw_loss(temporary):
        columns = list(metrics)
        cleaned = cache.path.parent / f"{name}_metrics.csv"
        lines = [",".join(columns)] + [",".join(str(metrics[column][row]) for column in columns) for row in range(len(metrics["loss"]))]
        cleaned.write_text("\n".join(lines) + "\n", encoding="utf-8")
        plot_losses(str(cleaned), temporary)
        cleaned.unlink(missing_ok=True)

    if metrics and len(metrics["loss"]):
        note(cache.ensure(f"loss:{name}", run["metrics_signature"], loss_path, lambda: render(loss_path, draw_loss)))
    else:
        note("loss curves: no metrics rows yet")
    figures["loss"] = loss_path if loss_path.exists() else None

    plot_logs, info = run["plot_logs"], run["info"]
    chain_path = out / f"{name}_chain.png"
    if plot_logs and len(plot_logs["final_var"]):
        signature = (f"{file_signature(run_dir / 'logs' / 'plot_logs.csv')} var={data_variance} mean={data_mean} "
                     f"num_iter={run['num_iter']} n_ipf={run['n_ipf']} v2")
        note(cache.ensure(f"chain:{name}", signature, chain_path, lambda: render(chain_path, lambda temporary: draw_chain_statistics(
            plot_logs, run["num_iter"], run["n_ipf"], data_variance, data_mean, info.get("mean_final"), info.get("var_final"), temporary))))
    else:
        note("chain statistics: no plot_logs rows yet")
    figures["chain"] = chain_path if chain_path.exists() else None

    if run["kind"] == "toy":
        return toy_run_figures(run, out, cache, figures, note)
    for direction in ("b", "f"):
        files = grid_files(run_dir, direction)
        latest_path, montage_path = out / f"{name}_latest_{direction}.png", out / f"{name}_montage_{direction}.png"
        if files:
            latest = files[-1]
            note(cache.ensure(f"latest:{name}:{direction}", file_signature(latest), latest_path,
                              lambda: render(latest_path, lambda temporary: shutil.copyfile(latest, temporary))))
            signature = json.dumps([file_signature(file) for file in files])
            note(cache.ensure(f"montage:{name}:{direction}", signature, montage_path,
                              lambda: render(montage_path, lambda temporary: montage(str(run_dir), temporary, directions=(direction,)))))
            figures[f"latest_name_{direction}"] = latest.parent.name
        else:
            note(f"no {DIRECTION_LABEL[direction]} sample grids yet")
        figures[f"latest_{direction}"] = latest_path if latest_path.exists() else None
        figures[f"montage_{direction}"] = montage_path if montage_path.exists() else None
    return figures


def copy_latest(run: dict, out: Path, cache: FigureCache, note, key: str, source: Path | None, target: Path) -> Path | None:
    if source is None:
        return target if target.exists() else None
    note(cache.ensure(f"{key}:{run['name']}", file_signature(source), target,
                      lambda: render(target, lambda temporary: shutil.copyfile(source, temporary))))
    return target if target.exists() else None


def draw_toy_data(run: dict, path: str):
    """The training points of a toy run, regenerated from its config, with the prior of info.json."""
    from .toy import draw_points, toy_points

    data, info = run["config"].get("data") or {}, run["info"]
    points = toy_points(data["distribution"], int(data.get("npar", 10000)), int(data.get("seed", 0)))
    sample = points[np.random.default_rng(1).choice(points.shape[0], min(1000, points.shape[0]), replace=False)]
    draw_points(path, sample, info.get("mean_final_vector", [0.0, 0.0]), float(info.get("var_final", 1.0)), "data, prior mean and sigma")


def toy_run_figures(run: dict, out: Path, cache: FigureCache, figures: dict, note) -> dict:
    name, run_dir = run["name"], run["dir"]
    data_path = out / f"{name}_data.png"
    if run["info"] and run["distribution"]:
        signature = json.dumps([run["config"].get("data"), run["info"].get("mean_final_vector"), run["info"].get("var_final")])
        note(cache.ensure(f"toydata:{name}", signature, data_path, lambda: render(data_path, lambda temporary: draw_toy_data(run, temporary))))
    figures["data"] = data_path if data_path.exists() else None
    for direction in ("b", "f"):
        for filename, kind in (("samples.png", "samples"), ("trajectories.png", "chains")):
            files = grid_files(run_dir, direction, filename)
            latest = files[-1] if files else None
            figures[f"{kind}_{direction}"] = copy_latest(run, out, cache, note, f"toy{kind}{direction}", latest,
                                                         out / f"{name}_{kind}_{direction}.png")
            if latest is not None:
                figures[f"latest_name_{direction}"] = latest.parent.name
        if not grid_files(run_dir, direction, "samples.png"):
            note(f"no {DIRECTION_LABEL[direction]} sample figures yet")
    return figures


# ---------------------------------------------------------------- toy distribution validation
TOY_VARIANT_LABELS = {"dense": "dense", "maskfull": "mask K=full", "mask32": "mask K=32", "gather32": "gather K=32"}


def toy_groups(toy_runs: list[dict]) -> list[tuple[str, list[dict]]]:
    """[(group, runs sorted by variant)], groups in the order of TOY_DISTRIBUTIONS; a group is the run name
    between toy_ and the variant suffix (toy_moon_dense -> moon)."""
    order = {name: index for index, name in enumerate(TOY_DISTRIBUTIONS)}
    variant_order = {name: index for index, name in enumerate(TOY_VARIANTS)}
    for run in toy_runs:
        run["group"] = run["name"].removeprefix("toy_").rsplit("_", 1)[0]
    names = sorted({run["group"] for run in toy_runs}, key=lambda name: (order.get(name.rstrip("centred"), 99), name))
    return [(name, sorted((run for run in toy_runs if run["group"] == name), key=lambda run: (variant_order.get(run["variant"], 99), run["name"])))
            for name in names]


def toy_group_targets(group: list[dict]):
    """(data_target, prior_target, prior mean, prior variance) of a group, from the first run's config and info."""
    from .toy import toy_points, toy_targets

    run = next((run for run in group if run["config"].get("data") and run["info"]), group[0])
    data, info = run["config"].get("data") or {}, run["info"]
    points = toy_points(data["distribution"], int(data.get("npar", 10000)), int(data.get("seed", 0)))
    mean, var = info.get("mean_final_vector", [0.0, 0.0]), float(info.get("var_final", 1.0))
    data_target, prior_target = toy_targets(points, mean, var, int(run["ipf"].get("plot_npar", 1000)))
    return data_target, prior_target, mean, var


def latest_samples(run: dict, direction: str) -> tuple[str, Path] | None:
    files = grid_files(run["dir"], direction, "samples.npz")
    return (files[-1].parent.name, files[-1]) if files else None


def logged_energy(run: dict, direction: str, tag: str) -> float | None:
    rows = run["toy_metrics"]
    if not rows or not len(rows["n"]):
        return None
    _, n, i = tag.split("_")
    mask = (rows["fb"] == direction) & (rows["n"] == int(n)) & (rows["i"] == int(i))
    if not mask.any():
        return None
    return float((rows["energy_to_data"] if direction == "b" else rows["energy_to_prior"])[mask][-1])


def _toy_axis(axis, limit=15.0):
    axis.set_xlim(-limit, limit); axis.set_ylim(-limit, limit); axis.set_aspect("equal")
    axis.set_xticks([]); axis.set_yticks([])


def draw_toy_matrix(groups: list[tuple[str, list[dict]]], direction: str, path: str):
    """Rows: groups (data in the first column), columns: the networks; each panel the final samples of the
    latest half-step of that direction over the target cloud, with the logged energy distance."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    columns = ["data"] + [TOY_VARIANT_LABELS[variant] for variant in TOY_VARIANTS]
    figure, axes = plt.subplots(len(groups), len(columns), figsize=(1.7 * len(columns), 1.7 * len(groups)), squeeze=False)
    for row, (name, group) in enumerate(groups):
        data_target, prior_target, mean, var = toy_group_targets(group)
        target = data_target if direction == "b" else prior_target
        axis = axes[row][0]
        axis.scatter(data_target[:, 0], data_target[:, 1], s=1.5, color="0.5", linewidths=0)
        axis.add_patch(plt.Circle((mean[0], mean[1]), var ** 0.5, fill=False, color="tab:orange", lw=0.8))
        axis.plot(mean[0], mean[1], "o", color="tab:orange", ms=2)
        axis.set_ylabel(f"{name}\nN(({mean[0]:.1f}, {mean[1]:.1f}), {var:.1f} I)", fontsize=7)
        by_variant = {run["variant"]: run for run in group}
        for column, variant in enumerate(TOY_VARIANTS, start=1):
            axis, run = axes[row][column], by_variant.get(variant)
            found = latest_samples(run, direction) if run else None
            if found is None:
                axis.text(0.5, 0.5, "no samples yet" if run else "no run", ha="center", va="center", fontsize=7, color="0.5", transform=axis.transAxes)
            else:
                tag, npz = found
                final = np.load(npz)["final"]
                axis.scatter(target[:, 0], target[:, 1], s=1.5, color="0.78", linewidths=0)
                axis.scatter(final[:, 0], final[:, 1], s=1.5, color="tab:blue", linewidths=0)
                energy, n = logged_energy(run, direction, tag), int(tag.split("_")[1])
                label = (f"{energy:.2f}" if energy is not None else "") + (f"  n={n}" if n != run["n_ipf"] else "")
                axis.text(0.04, 0.96, label, transform=axis.transAxes, fontsize=7, va="top")
        for column in range(len(columns)):
            _toy_axis(axes[row][column])
    for column, label in enumerate(columns):
        axes[0][column].set_title(label, fontsize=8)
    figure.subplots_adjust(left=0.075, right=0.995, top=0.96, bottom=0.005, wspace=0.04, hspace=0.04)
    figure.savefig(path, dpi=100)
    plt.close(figure)


def draw_toy_energy_strip(groups: list[tuple[str, list[dict]]], path: str):
    """One panel per group: the energy distance of the backward samples to the data along the IPF iterations."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    variant_order = {name: index for index, name in enumerate(TOY_VARIANTS)}
    figure, axes = plt.subplots(1, len(groups), figsize=(2.3 * len(groups), 2.3), squeeze=False)
    for axis, (name, group) in zip(axes[0], groups):
        for run in group:
            rows = run["toy_metrics"]
            if not rows or not len(rows["n"]):
                continue
            backward = rows["fb"] == "b"
            if backward.any():
                axis.plot(rows["n"][backward], rows["energy_to_data"][backward], marker="o", ms=2, lw=1,
                          color=f"C{variant_order.get(run['variant'], 9)}", label=TOY_VARIANT_LABELS.get(run["variant"], run["variant"]))
        axis.set_title(name, fontsize=8); axis.set_yscale("log"); axis.grid(alpha=0.3)
        axis.xaxis.set_major_locator(MaxNLocator(integer=True)); axis.tick_params(labelsize=7)
        axis.set_xlabel("IPF iteration", fontsize=7)
    axes[0][0].set_ylabel("energy distance to data", fontsize=7)
    if axes[0][0].get_legend_handles_labels()[0]:
        axes[0][0].legend(fontsize=6)
    figure.tight_layout(pad=0.4)
    figure.savefig(path, dpi=100)
    plt.close(figure)


def toy_tab(runs: list[dict], figures_of: dict, out: Path, cache: FigureCache) -> str:
    toy_runs = [run for run in runs if run["kind"] == "toy"]
    parts = ["<h2>Toy distribution validation</h2>",
             "<p class='caption'>Rows: the distributions of bridge/data/two_dim.py with their priors (orange: mean and one sigma); columns: "
             "the reference score network (dense) and the same network with its linear layers replaced by sparse-transformers layers "
             "(st's masked SparseLinear with the full support or fan-in 32, z1t's gather SparseLinear with fan-in 32). Each panel: the "
             "1000 final backward samples (blue) of the latest half-step over 1000 data points (grey); the number is their energy "
             "distance, 0 for equal distributions. Settings of conf/dataset/2d.yaml: 20 steps of gamma 0.01, 20 IPF iterations of "
             "10 000 steps, batch 512, drift matching, no EMA, a fresh network per half-step; the reference prior is the mean and "
             "variance of the first 100 points, which for the mixture all lie in the left Gaussian; mixturecentred uses the N(0, I) "
             "prior written in the same yaml.</p>"]
    if not toy_runs:
        return "".join(parts + ["<p class='muted'>no toy runs yet (bash slurm/toy_runs.sh)</p>"])
    groups = toy_groups(toy_runs)
    signature = json.dumps([[run["name"], file_signature(run["dir"] / "logs" / "toy_metrics.csv"),
                             [str(latest_samples(run, direction)) for direction in "bf"], run["info"].get("var_final")]
                            for _, group in groups for run in group])
    for direction, width in (("b", 850), ("f", 850)):
        matrix = out / f"toy_matrix_{direction}.png"
        error = cache.ensure(f"toymatrix:{direction}", signature, matrix,
                             lambda: render(matrix, lambda temporary: draw_toy_matrix(groups, direction, temporary)))
        if error:
            parts.append(f"<p class='warn'>{escape_html(error)}</p>")
    strip = out / "toy_energy.png"
    error = cache.ensure("toyenergy", signature, strip, lambda: render(strip, lambda temporary: draw_toy_energy_strip(groups, temporary)))
    if error:
        parts.append(f"<p class='warn'>{escape_html(error)}</p>")
    matrix_b, matrix_f = out / "toy_matrix_b.png", out / "toy_matrix_f.png"
    parts.append(figure(matrix_b if matrix_b.exists() else None, "backward samples (prior to data) over the data", 850))
    parts.append(figure(strip if strip.exists() else None, "energy distance of the backward samples to the data along the IPF iterations", 1150))
    header = ["distribution"] + [TOY_VARIANT_LABELS[variant] for variant in TOY_VARIANTS]
    rows = []
    for name, group in groups:
        by_variant = {run["variant"]: run for run in group}
        cells = [escape_html(name)]
        for variant in TOY_VARIANTS:
            run = by_variant.get(variant)
            if run is None:
                cells.append("")
                continue
            metrics, latest = run["toy_metrics"], run["latest"]
            text = ""
            if metrics and len(metrics["n"]) and (metrics["fb"] == "b").any():
                backward = metrics["fb"] == "b"
                values, iterations = metrics["energy_to_data"][backward], metrics["n"][backward]
                best = int(np.argmin(values))
                text = f"{values[-1]:.2f} (best {values[best]:.2f} at n={int(iterations[best])})"
            if run["status"] != "finished":
                text += f" <span class='{run['status_class']}'>{escape_html(run['status'])}" + (f", {latest['fb']} n={latest['n']}" if latest else "") + "</span>"
            cells.append(f"<a href='#run-{escape_html(run['name'])}'>{text or 'no samples yet'}</a>")
        rows.append(cells)
    parts.append("<p class='caption'>Energy distance of the backward samples to the data after the last half-step, and the best iteration.</p>" + table(header, rows))
    parts.append("<details><summary>Forward direction: final samples of the forward half-steps (data to prior) over prior samples</summary>" +
                 figure(matrix_f if matrix_f.exists() else None, "forward samples over 1000 prior samples; the number is the energy distance to the prior", 850) +
                 "</details>")
    return "".join(parts)


# ---------------------------------------------------------------- data
def draw_data_grid(grid: Path) -> dict:
    array = np.load(DATA_ARRAY, mmap_mode="r")
    generator = np.random.default_rng(0)
    grid_indices = np.sort(generator.choice(array.shape[0], 100, replace=False))
    images = np.asarray(array[grid_indices]).astype(np.float32) / 255.0
    render(grid, lambda temporary: save_image_grid(images.transpose(0, 3, 1, 2), temporary))
    sample_indices = np.sort(generator.choice(array.shape[0], 1000, replace=False))
    sample = np.asarray(array[sample_indices]).astype(np.float32) / 255.0
    return {"shape": list(array.shape), "dtype": str(array.dtype), "mean": float(sample.mean()), "variance": float(sample.var()),
            "channel_means": [float(value) for value in sample.mean(axis=(0, 1, 2))], "sample": int(sample.shape[0])}


def data_tab(out: Path, cache: FigureCache) -> tuple[str, float | None]:
    grid = out / "data_grid.png"
    signature = f"{SCRIPT_STAMP} {file_signature(DATA_ARRAY)}"
    statistics, notes = cache.entries.get("data"), []
    if not (isinstance(statistics, dict) and statistics.get("signature") == signature and grid.exists()):
        try:
            statistics = draw_data_grid(grid)
            statistics["signature"] = signature
            cache.entries["data"] = statistics
            cache.redrawn += 1
        except Exception as error:
            statistics = cache.entries.get("data") if isinstance(cache.entries.get("data"), dict) else None
            notes.append(f"data grid: {type(error).__name__}: {error}")
    facts = read_json(DATA_FACTS)
    rows = [(key, value) for key, value in facts.items()]
    if DATA_ARRAY.exists():
        rows.append(("array", f"{DATA_ARRAY.name}, {DATA_ARRAY.stat().st_size / 2**30:.2f} GiB, written {stamp(DATA_ARRAY)}"))
    else:
        notes.append(f"{DATA_ARRAY} missing")
    if statistics:
        rows += [("shape, dtype", f"{tuple(statistics['shape'])}, {statistics['dtype']}"),
                 (f"pixel mean, variance ({statistics['sample']} images in [0, 1])", f"{statistics['mean']:.4f}, {statistics['variance']:.4f}"),
                 ("channel means (R, G, B)", ", ".join(f"{value:.3f}" for value in statistics["channel_means"]))]
    parts = ["<h2>Training data</h2>",
         "<p class='caption'>100 training images drawn with seed 0 from the memory-mapped array (the same images every cycle), "
         "and the dataset facts of data/celeba64_train.json. The pixel variance of a fixed 1,000-image sample is the target "
         "drawn as the data variance in the chain-statistics figures; the runs' prior mean is the mean of the first 100 images (info.json).</p>"]
    parts += [f"<p class='warn'>{escape_html(text)}</p>" for text in notes]
    parts.append("<div class='cards'>" + figure(grid if grid.exists() else None, "100 training images, seed 0", 662) +
             "<table>" + "".join(f"<tr><th>{escape_html(key)}</th><td>{escape_html(value)}</td></tr>" for key, value in rows) + "</table></div>")
    return "".join(parts), statistics["variance"] if statistics else None, statistics["mean"] if statistics else None


# ---------------------------------------------------------------- html
CSS = """body{font-family:system-ui,sans-serif;margin:18px;max-width:1500px;color:#222}
table{border-collapse:collapse;font-size:12.5px}td,th{border:1px solid #ccc;padding:3px 6px;text-align:left;vertical-align:top}
th{background:#f2f2f2}code{font-size:12px}pre{font-size:12px;background:#f7f7f7;border:1px solid #ddd;padding:6px 8px;overflow-x:auto;max-width:1100px}
.muted{color:#777}img{max-width:100%;border:1px solid #ddd}details.run{border:1px solid #ddd;border-radius:4px;padding:2px 10px 8px;margin:10px 0}
summary{cursor:pointer;font-size:15px;padding:6px 0}h2{margin-top:28px;border-bottom:1px solid #ddd}h4{margin:14px 0 4px;font-size:14px}
.ok{color:#176b2d}.bad{color:#a40000}.warn{color:#9a6700}.cards{display:flex;flex-wrap:wrap;align-items:flex-start;gap:8px}
figure{margin:4px;display:inline-block;vertical-align:top}figcaption{font-size:11.5px;color:#444}p.caption{font-size:12.5px;color:#444;max-width:1100px}
table.config td{padding:1px 6px}table.config td.muted{color:#777}
nav.tabs{border-bottom:1px solid #ccc;margin:12px 0 0}nav.tabs button{font:inherit;font-size:14px;background:none;border:1px solid #ccc;border-bottom:none;padding:6px 14px;margin-right:4px;cursor:pointer;color:#555}
nav.tabs button.active{color:#222;background:#f2f2f2}section.tab{display:none}section.tab.active{display:block}"""

SCRIPT = """(function () {
  // The page reloads itself every RELOAD_MS. Across reloads it keeps the chosen tab (hash and localStorage),
  // which run sections are open (localStorage, keyed by the section id) and the scroll position (sessionStorage).
  var openKey = 'sparse_dsb_dashboard_open', scrollKey = 'sparse_dsb_dashboard_scroll';
  var state = {};
  try { state = JSON.parse(localStorage.getItem(openKey) || '{}') || {}; } catch (e) { state = {}; }
  var sections = document.querySelectorAll('details[id]');
  for (var d = 0; d < sections.length; d++) {
    if (Object.prototype.hasOwnProperty.call(state, sections[d].id)) { sections[d].open = !!state[sections[d].id]; }
    sections[d].addEventListener('toggle', function () {
      state[this.id] = this.open;
      try { localStorage.setItem(openKey, JSON.stringify(state)); } catch (e) {}
    });
  }
  try { if ('scrollRestoration' in history) { history.scrollRestoration = 'manual'; } } catch (e) {}
  if (RELOAD_MS > 0) {
    window.setTimeout(function () {
      try { sessionStorage.setItem(scrollKey, String(window.pageYOffset)); } catch (e) {}
      window.location.reload();
    }, RELOAD_MS);
  }
  var key = 'sparse_dsb_dashboard_tab', fallback = 'runs';
  var buttons = document.querySelectorAll('nav.tabs button'), panels = document.querySelectorAll('section.tab');
  function known(name) { return !!(name && document.getElementById('tab-' + name)); }
  function show(name) {
    for (var i = 0; i < panels.length; i++) { panels[i].className = panels[i].id === 'tab-' + name ? 'tab active' : 'tab'; }
    for (var j = 0; j < buttons.length; j++) { buttons[j].className = buttons[j].getAttribute('data-tab') === name ? 'active' : ''; }
  }
  function choose() {
    var name = window.location.hash ? window.location.hash.substring(1) : '';
    if (name.indexOf('run-') === 0) {
      // a run link: open and show that section once, then drop the hash so later reloads keep the scroll position instead
      show('runs'); var target = document.getElementById(name);
      if (target) {
        target.open = true; target.scrollIntoView();
        state[name] = true;
        try { localStorage.setItem(openKey, JSON.stringify(state)); } catch (e) {}
      }
      try { history.replaceState(null, '', '#runs'); } catch (e) {}
      return true;
    }
    if (!known(name)) { try { name = localStorage.getItem(key); } catch (e) { name = ''; } }
    show(known(name) ? name : fallback);
  }
  var jumped = choose();
  if (!jumped) {
    try {
      var y = sessionStorage.getItem(scrollKey);
      if (y !== null) { sessionStorage.removeItem(scrollKey); window.scrollTo(0, parseInt(y, 10) || 0); }
    } catch (e) {}
  }
  for (var k = 0; k < buttons.length; k++) {
    buttons[k].addEventListener('click', function () {
      var name = this.getAttribute('data-tab');
      try { localStorage.setItem(key, name); } catch (e) {}
      try { history.replaceState(null, '', '#' + name); } catch (e) {}
      show(name);
    });
  }
  window.addEventListener('hashchange', choose);
})();"""


def figure(path: Path | None, caption: str, width: int) -> str:
    if path is None:
        return f"<figure style='width:{width}px'><figcaption class='muted'>{escape_html(caption)}: not available yet</figcaption></figure>"
    with Image.open(path) as image:
        pixels_wide, pixels_high = image.size
    height = round(width * pixels_high / pixels_wide)
    return (f"<figure style='width:{width}px'><a href='{escape_html(path.name)}'><img src='{escape_html(path.name)}' alt='{escape_html(caption)}' "
            f"width='{width}' height='{height}' style='width:{width}px;height:auto'></a>"
            f"<figcaption>{escape_html(caption)}</figcaption></figure>")


def table(header: list[str], rows: list[list[str]], css_class: str = "") -> str:
    head = "".join(f"<th>{cell}</th>" for cell in header)
    body = "".join("<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows)
    attribute = f" class='{css_class}'" if css_class else ""
    return f"<table{attribute}><tr>{head}</tr>{body}</table>"


def config_table(config: dict) -> str:
    items = [(f"{section}.{key}", value) for section in ("ipf", "model") for key, value in (config.get(section) or {}).items()]
    columns = 3
    height = (len(items) + columns - 1) // columns
    rows = []
    for row in range(height):
        cells = ""
        for column in range(columns):
            index = column * height + row
            cells += f"<td class='muted'>{escape_html(items[index][0])}</td><td>{escape_html(items[index][1])}</td>" if index < len(items) else "<td></td><td></td>"
        rows.append(f"<tr>{cells}</tr>")
    return "<table class='config'>" + "".join(rows) + "</table>"


def run_row(run: dict) -> list[str]:
    model, info, latest = run["model"], run["info"], run["latest"]
    parameters = f"{info['parameters_stored']:,} / {info['parameters_active']:,}" if "parameters_stored" in info else ""
    progress = f"<br><span class='muted'>{100 * latest['progress']:.0f}% of the run</span>" if latest and latest["progress"] is not None else ""
    return [f"<a href='#run-{escape_html(run['name'])}'>{escape_html(run['name'])}</a>",
            escape_html(model_label(run)), escape_html(model.get("n_layers", "")), escape_html(model.get("n_embed", "")),
            escape_html(model.get("linear_fan_in", "")), escape_html(model.get("patch_size", "")), parameters,
            f"{run['n_ipf']} x {run['num_iter']:,}" if run["num_iter"] else "",
            f"<span class='{run['status_class']}'>{escape_html(run['status'])}</span><br><span class='muted'>{escape_html(run['status_detail'])}</span>",
            f"{latest['fb']} n={latest['n']}" if latest else "",
            f"{latest['i']:,} / {run['num_iter']:,}{progress}" if latest else "",
            f"{latest['loss']:.4f}" if latest else "", f"{latest['last_10pct_loss']:.4f}" if latest else "",
            duration(run["elapsed"]), f"{latest['rate']:.1f}" if latest and latest["rate"] else ""]


def run_section(run: dict, figures: dict) -> str:
    name, model, latest = run["name"], run["model"], run["latest"]
    position = f"{latest['fb']} n={latest['n']} i={latest['i']:,}" if latest else "no metrics"
    if run["kind"] == "toy":
        description = f"{escape_html(run['distribution'])}, score network {escape_html(model_label(run))}"
    else:
        description = (f"{escape_html(model.get('family', '?'))}, {escape_html(model.get('n_layers', '?'))} blocks, "
                       f"width {escape_html(model.get('n_embed', '?'))}, K {escape_html(model.get('linear_fan_in', '?'))}")
    summary = (f"{escape_html(name)} <span class='muted'>{description}; "
               f"<span class='{run['status_class']}'>{escape_html(run['status'])}</span>, {escape_html(position)}</span>")
    opened = " open" if run["status"] in ("running", "running?", "queued") else ""
    parts = [f"<details class='run' id='run-{escape_html(name)}'{opened}><summary>{summary}</summary>"]
    parts += [f"<p class='warn'>{escape_html(text)}</p>" for text in run["notes"]]
    parts.append("<h4>Mean-matching loss</h4><p class='caption'>Left: every logged step of the run, blue backward and orange forward half-steps. "
             "Right: one curve per half-step over its own steps, darker for later IPF iterations.</p>" +
             figure(figures.get("loss"), f"{name}: loss curves (logs/metrics.csv)", 1100))
    parts.append("<h4>Chain statistics</h4><p class='caption'>Final variance (log scale) and final mean of the 100 sampled chains at every "
             "checkpoint, against the half-step: backward chains start from the prior and should reach the data statistics, forward chains "
             "start from data and should reach the prior; ref is the reference Ornstein-Uhlenbeck chain before training.</p>" +
             figure(figures.get("chain"), f"{name}: final variance and mean of the sampled chains (logs/plot_logs.csv)", 1100))
    if run["kind"] == "toy":
        cards = [figure(figures.get("data"), "data and prior", 300)]
        cards += [figure(figures.get(f"{kind}_{direction}"), f"{figures.get(f'latest_name_{direction}', direction)}: {DIRECTION_LABEL[direction]}, {what}", 300)
                  for direction in "bf" for kind, what in (("samples", "final samples"), ("chains", "chains"))]
        parts.append(f"<h4>Latest samples</h4><div class='cards'>{''.join(cards)}</div>")
    else:
        cards = "".join(figure(figures.get(f"latest_{direction}"),
                               f"{figures.get(f'latest_name_{direction}', direction)}: {DIRECTION_LABEL[direction]}, final samples", 440)
                        for direction in "bf")
        parts.append(f"<h4>Latest sample grids</h4><div class='cards'>{cards}</div>")
        cards = "".join(figure(figures.get(f"montage_{direction}"), f"{DIRECTION_LABEL[direction]}: first row of the final grid at the last 8 checkpoints", 662)
                        for direction in "bf")
        parts.append(f"<h4>Sample grids along the half-steps</h4><div class='cards'>{cards}</div>")
    if run["half_steps"]:
        rows = [[str(row["n"]), row["fb"], f"{row['first_loss']:.5f}", f"{row['last_10pct_loss']:.5f}", f"{row['min_loss']:.5f}",
                 f"{row['iterations']:,}", f"{row['seconds']:.0f}"] for row in run["half_steps"]]
        parts.append("<h4>Half-steps</h4>" + table(["n", "fb", "first loss", "last-10% loss", "min loss", "steps logged", "seconds"], rows))
    if run["config"]:
        parts.append("<h4>Config</h4>" + config_table(run["config"]))
    if run["log_tail"]:
        parts.append(f"<h4>Job log</h4><p class='muted'>{escape_html(run['latest_log'].relative_to(PROJECT))}, {escape_html(stamp(run['latest_log']))}</p>"
                 f"<pre>{escape_html(chr(10).join(run['log_tail']))}</pre>")
    parts.append("</details>")
    return "".join(parts)


def queue_tab(jobs: list[dict], squeue_error: str, log_runs: dict[Path, Path], cycle_note: str) -> str:
    parts = ["<h2>Slurm queue</h2>", f"<p class='caption'>{escape_html(cycle_note)}</p>"]
    if squeue_error:
        parts.append(f"<p class='bad'>{escape_html(squeue_error)}</p>")
    rows = [[escape_html(job["id"]), escape_html(job["name"]), escape_html(job["state"]), escape_html(job["time"]), escape_html(job["limit"]), escape_html(job["partition"]),
             escape_html(job["nodes"] or job["reason"]), escape_html(log_runs[job["log"]].name) if job["log"] in log_runs else ""] for job in jobs]
    parts.append(table(["job", "name", "state", "time", "limit", "partition", "node or reason", "run"], rows) if rows else "<p class='muted'>no jobs</p>")
    return "".join(parts)


def benchmark_tab() -> str:
    try:
        text, when = BENCHMARK.read_text(encoding="utf-8"), stamp(BENCHMARK)
    except OSError as error:
        return f"<h2>Benchmark</h2><p class='warn'>{escape_html(BENCHMARK.relative_to(PROJECT))}: {escape_html(error)}</p>"
    return f"<h2>Benchmark</h2><p class='caption'>{escape_html(BENCHMARK.relative_to(PROJECT))} verbatim, written {escape_html(when)}.</p><pre>{escape_html(text)}</pre>"


def generate(out: Path, state: Path, cycle: int, loop_seconds: int) -> str:
    started = time.time()
    out.mkdir(parents=True, exist_ok=True)
    cache = FigureCache(state)
    jobs, squeue_error = slurm_jobs()
    log_runs = {}
    for log in SLURM_LOGS.glob("*.out") if SLURM_LOGS.is_dir() else []:
        directory = run_dir_of_log(log)
        if directory is not None:
            log_runs[log] = directory
    data_html, data_variance, data_mean = data_tab(out, cache)
    runs = [scan_run(run_dir.resolve(), jobs, log_runs, squeue_error)
            for run_dir in sorted(RUNS.iterdir()) if run_dir.is_dir()] if RUNS.is_dir() else []
    figures_of = {run["name"]: run_figures(run, out, cache, data_variance, data_mean) for run in runs}
    sections = [run_section(run, figures_of[run["name"]]) for run in runs]
    toy_html = toy_tab(runs, figures_of, out, cache)
    now = utc_now()
    seconds = time.time() - started
    refresh = f"the data refresh every {loop_seconds} s and the page reloads with them" if loop_seconds else "generated once; rerun to refresh"
    cycle_note = (f"Dashboard cycle {cycle} on {socket.gethostname()} (Slurm job {os.environ.get('SLURM_JOB_ID', 'none')}) "
                  f"regenerated the page at {now} in {seconds:.1f} s; {refresh}.")
    parts = [f"<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'><title>Schrödinger distillation dashboard</title><style>{CSS}</style><noscript><style>section.tab{{display:block}}</style></noscript></head><body>",
         f"<h1>Schrödinger distillation: IPF on sparse-transformers models</h1><p class='muted'>Generated {now}; {refresh}; the tab, the open run sections and the scroll position are kept.</p>",
         "<nav class='tabs'>" + "".join(f"<button type='button' data-tab='{name}'>{escape_html(label)}</button>" for name, label in TABS) + "</nav>",
         "<section class='tab' id='tab-runs'><h2>Runs</h2>",
         "<p class='caption'>One row per directory under runs/. K is the fan-in of the sparse linear layers; parameters are stored / active (info.json); "
         "IPF is iterations x steps per half-step; the last-10% loss and the steps per second are over the last 10% of the logged steps of the "
         "latest half-step (cache refreshes included); elapsed is done.json for finished runs and the age of metrics.csv relative to config.yaml otherwise.</p>",
         table(["run", "family", "blocks", "width", "K", "patch", "parameters", "IPF", "status", "half-step", "step", "latest loss",
                "last-10% loss", "elapsed", "steps/s"], [run_row(run) for run in runs]) if runs else "<p class='muted'>no runs</p>",
         *sections, "</section>",
         f"<section class='tab' id='tab-toy'>{toy_html}</section>",
         f"<section class='tab' id='tab-data'>{data_html}</section>",
         f"<section class='tab' id='tab-queue'>{queue_tab(jobs, squeue_error, log_runs, cycle_note)}</section>",
         f"<section class='tab' id='tab-benchmark'>{benchmark_tab()}</section>",
         f"<p class='muted'>Generated by sparse_dsb/dashboard.py at {now}.</p>",
         f"<script>var RELOAD_MS = {1000 * loop_seconds};\n{SCRIPT}</script></body></html>"]
    write_atomic(out / "index.html", "\n".join(parts))
    cache.save()
    running = sum(run["status"] == "running" for run in runs)
    notes = sum(len(run["notes"]) for run in runs)
    return (f"{now} cycle {cycle}: {len(runs)} runs ({running} running, {len(jobs)} jobs in queue), "
            f"{cache.redrawn} figures redrawn, {notes} notes, {time.time() - started:.1f} s")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="dashboard", help="output directory (default dashboard/)")
    parser.add_argument("--state", default=".scratch/dashboard_state",
                        help="directory for the figure cache and scratch files, outside the served directory")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--once", action="store_true", help="one cycle (the default)")
    group.add_argument("--loop", type=int, default=0, metavar="SECONDS", help="regenerate every SECONDS")
    args = parser.parse_args(argv)
    out, state = Path(args.out).resolve(), Path(args.state).resolve()
    cycle = 0
    while True:
        cycle += 1
        started = time.time()
        try:
            print(generate(out, state, cycle, args.loop), flush=True)
        except Exception:
            print(f"{utc_now()} cycle {cycle} failed:\n{traceback.format_exc()}", flush=True)
        if not args.loop:
            break
        time.sleep(max(1.0, args.loop - (time.time() - started)))


if __name__ == "__main__":
    main()
