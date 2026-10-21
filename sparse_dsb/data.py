"""CelebA for the bridge: a one-off preprocessing pass and a device-side sampler.

Preprocessing reproduces the reference transform (center crop 140, resize to
image_size with bilinear filtering, pixels in [0, 1]) for the official train
split and stores it as a uint8 array of shape (N, image_size, image_size, 3).
Reading the original zip once is far cheaper over NFS than 162k small files.

    python -m sparse_dsb.data --zip <img_align_celeba.zip> --partition <list_eval_partition.txt> \
        --out data/celeba64_train.npy --image-size 64 --workers 16

The sampler draws a random batch on the device and applies the random
horizontal flip of the reference data loader. Batches are float32 in [0, 1] with
layout (B, C, H, W).
"""
from __future__ import annotations

import argparse
import io
import json
import os
import time
import zipfile
from multiprocessing import Pool

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image

CROP = 140
_zip = None


def train_split_names(partition_path: str, split: int = 0) -> list[str]:
    names = []
    with open(partition_path) as handle:
        for line in handle:
            name, part = line.split()
            if int(part) == split:
                names.append(name)
    return sorted(names)


def _open_zip(zip_path):
    global _zip
    _zip = zipfile.ZipFile(zip_path)


def _process(args):
    member, image_size = args
    with _zip.open(member) as handle:
        image = Image.open(io.BytesIO(handle.read())).convert("RGB")
    width, height = image.size
    left, top = int(round((width - CROP) / 2.0)), int(round((height - CROP) / 2.0))
    image = image.crop((left, top, left + CROP, top + CROP))
    image = image.resize((image_size, image_size), Image.BILINEAR)
    return np.asarray(image, dtype=np.uint8)


def build_array(zip_path: str, partition_path: str, out_path: str, image_size: int, workers: int, split: int = 0, limit: int | None = None):
    names = train_split_names(partition_path, split)
    if limit:
        names = names[:limit]
    with zipfile.ZipFile(zip_path) as archive:
        prefix = next(n for n in archive.namelist() if n.endswith(names[0])).rsplit("/", 1)[0]
    members = [(f"{prefix}/{name}", image_size) for name in names]
    start = time.time()
    with Pool(workers, initializer=_open_zip, initargs=(zip_path,)) as pool:
        images = np.stack(pool.map(_process, members, chunksize=256))
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    np.save(out_path, images)
    record = {
        "zip": zip_path, "partition": partition_path, "split": split, "count": len(names),
        "order": "sorted file name", "image_size": image_size,
        "recipe": f"RGB, center crop {CROP} (torchvision CenterCrop), PIL bilinear resize, uint8 NHWC",
        "seconds": round(time.time() - start, 1),
    }
    with open(os.path.splitext(out_path)[0] + ".json", "w") as handle:
        json.dump(record, handle, indent=2)
    return images, record


def load_images(path: str) -> np.ndarray:
    images = np.load(path, mmap_mode="r")
    if images.dtype != np.uint8 or images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError(f"expected uint8 NHWC array in {path}, got {images.dtype} {images.shape}")
    return images


def sample_images(key, images_uint8, batch_size: int, random_flip: bool = True):
    """Random batch as float32 in [0, 1], layout (B, C, H, W), with random horizontal flips."""
    index_key, flip_key = jax.random.split(key)
    index = jax.random.randint(index_key, (batch_size,), 0, images_uint8.shape[0])
    x = images_uint8[index].astype(jnp.float32) / 255.0
    x = jnp.transpose(x, (0, 3, 1, 2))
    if random_flip:
        flip = jax.random.bernoulli(flip_key, 0.5, (batch_size,))
        x = jnp.where(flip[:, None, None, None], x[..., ::-1], x)
    return x


def adaptive_mean(images_uint8, count: int = 100) -> float:
    """Mean pixel value of the first `count` images, the adaptive prior mean of the reference code."""
    return float(np.asarray(images_uint8[:count], dtype=np.float64).mean() / 255.0)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--zip", required=True)
    parser.add_argument("--partition", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    images, record = build_array(args.zip, args.partition, args.out, args.image_size, args.workers, limit=args.limit)
    print(json.dumps(record, indent=2), flush=True)
    print("shape", images.shape, "mean", images.mean() / 255.0, flush=True)


if __name__ == "__main__":
    main()
