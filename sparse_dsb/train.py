"""Run IPF on a sparse-transformers model.

    python -m sparse_dsb.train --config configs/celeba64_smoke.yaml [--out runs/name] [key=value ...]

The yaml has three sections: `ipf:` (IPFConfig), `model:` and `data:`. With
`data.kind: images` (the default) `data.path` names the preprocessed uint8 array
and `model:` is a ModelConfig; with `data.kind: toy`, `data.distribution` names
one of the toy distributions of sparse_dsb/toy.py, `data.npar` and `data.seed`
draw its points, and `model:` is a ToyModelConfig. Trailing key=value overrides
address fields as ipf.num_iter=200, model.n_layers=2 or data.distribution=moon.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time

import yaml

from .data import load_images
from .ipf import IPF, IPFConfig
from .models import ModelConfig
from .toy import ToyModelConfig, toy_points


def parse_overrides(pairs: list[str]) -> dict:
    overrides = {}
    for pair in pairs:
        key, _, value = pair.partition("=")
        section, _, field = key.partition(".")
        if not field:
            raise ValueError(f"override {pair!r} must look like section.field=value")
        overrides.setdefault(section, {})[field] = yaml.safe_load(value)
    return overrides


def build_configs(cfg: dict):
    ipf_config = IPFConfig(**cfg.get("ipf", {}))
    if cfg.get("data", {}).get("kind", "images") == "toy":
        model_config = ToyModelConfig(**cfg.get("model", {}))
    else:
        model_config = ModelConfig(**cfg.get("model", {}))
    return ipf_config, model_config


def load_data(data: dict):
    if data.get("kind", "images") == "toy":
        points = toy_points(data["distribution"], int(data.get("npar", 10000)), int(data.get("seed", 0)))
        print(f"{points.shape[0]} points of {data['distribution']} (seed {data.get('seed', 0)})", flush=True)
        return points
    images = load_images(data["path"])
    print(f"images {images.shape} from {data['path']}", flush=True)
    return images


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", default=None, help="output directory (default runs/<name>)")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args(argv)

    with open(args.config) as handle:
        cfg = yaml.safe_load(handle) or {}
    for section, fields in parse_overrides(args.overrides).items():
        cfg.setdefault(section, {}).update(fields)
    ipf_config, model_config = build_configs(cfg)
    name = cfg.get("name", os.path.splitext(os.path.basename(args.config))[0])
    out_dir = args.out or os.path.join("runs", name)
    if os.path.exists(os.path.join(out_dir, "logs", "metrics.csv")) and not ipf_config.checkpoint_run:
        print(f"{out_dir} already holds a run; pick another --out or resume with ipf.checkpoint_run=true", file=sys.stderr)
        sys.exit(2)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.yaml"), "w") as handle:
        yaml.safe_dump({"name": name, "data": cfg.get("data", {}), "ipf": dataclasses.asdict(ipf_config),
                        "model": dataclasses.asdict(model_config)}, handle, sort_keys=False)

    import jax
    print(f"jax {jax.__version__} backend={jax.default_backend()} devices={jax.devices()}", flush=True)
    if jax.default_backend() != "gpu":
        print("refusing to train without a GPU", file=sys.stderr, flush=True)
        sys.exit(1)

    data = load_data(cfg.get("data", {}))
    start = time.time()
    ipf = IPF(ipf_config, model_config, data, out_dir)
    ipf.train()
    with open(os.path.join(out_dir, "done.json"), "w") as handle:
        json.dump({"seconds": round(time.time() - start, 1)}, handle)
    print(f"done in {time.time() - start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
