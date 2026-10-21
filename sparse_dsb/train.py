"""Run IPF on a sparse-transformers model.

    python -m sparse_dsb.train --config configs/celeba64_smoke.yaml [--out runs/name] [key=value ...]

The yaml has three sections: `ipf:` (IPFConfig), `model:` (ModelConfig) and
`data:` (path of the preprocessed uint8 array). Trailing key=value overrides
address fields as ipf.num_iter=200 or model.n_layers=2.
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


def parse_overrides(pairs: list[str]) -> dict:
    overrides = {}
    for pair in pairs:
        key, _, value = pair.partition("=")
        section, _, field = key.partition(".")
        if not field:
            raise ValueError(f"override {pair!r} must look like section.field=value")
        overrides.setdefault(section, {})[field] = yaml.safe_load(value)
    return overrides


def build_configs(cfg: dict) -> tuple[IPFConfig, ModelConfig]:
    ipf_config = IPFConfig(**cfg.get("ipf", {}))
    model_config = ModelConfig(**cfg.get("model", {}))
    return ipf_config, model_config


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
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.yaml"), "w") as handle:
        yaml.safe_dump({"name": name, "data": cfg.get("data", {}), "ipf": dataclasses.asdict(ipf_config),
                        "model": dataclasses.asdict(model_config)}, handle, sort_keys=False)

    import jax
    print(f"jax {jax.__version__} backend={jax.default_backend()} devices={jax.devices()}", flush=True)
    if jax.default_backend() != "gpu":
        print("refusing to train without a GPU", file=sys.stderr, flush=True)
        sys.exit(1)

    images = load_images(cfg["data"]["path"])
    print(f"images {images.shape} from {cfg['data']['path']}", flush=True)
    start = time.time()
    ipf = IPF(ipf_config, model_config, images, out_dir)
    ipf.train()
    with open(os.path.join(out_dir, "done.json"), "w") as handle:
        json.dump({"seconds": round(time.time() - start, 1)}, handle)
    print(f"done in {time.time() - start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
