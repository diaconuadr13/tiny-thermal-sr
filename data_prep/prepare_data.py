"""Prepare the thermal SR dataset.

FLIR-IISR HR frames (1024x768 RGB) -> grayscale -> whole-frame resize to
64x48 (HR) -> x2 downsample to 32x24 (LR). Whole-frame resize keeps
full-scene FOV statistics comparable to a native low-resolution thermal
camera (MLX90640), unlike texture crops from the full-resolution image.

Also copies a fixed set of MLX90640 native 32x24 frames for qualitative
demo figures and firmware test inputs (no ground truth exists for these).

Source datasets live in the thesis repo (sisr-embedded-systems), not here.
Point --thesis-repo (or CAS_THESIS_REPO env var) at its checkout.

Usage:
    python data_prep/prepare_data.py [--thesis-repo /path/to/sisr-embedded-systems]
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_THESIS_REPO = os.environ.get(
    "CAS_THESIS_REPO", "/scratch/users/adiaconu/sisr-embedded-systems")

HR_SIZE = (64, 48)  # (width, height)
LR_SIZE = (32, 24)

# Fixed demo frames spread across the MLX90640 recording sessions.
MLX_DEMO_FRAMES = [0, 150, 320, 500, 700, 900, 1100, 1300, 1500, 1700]


def prepare_flir(out_root: Path, thesis_repo: Path) -> None:
    flir_splits = {
        "train": thesis_repo / "data" / "flir_iisr" / "train" / "HR",
        "val": thesis_repo / "data" / "flir_iisr" / "val" / "HR",
    }
    for split, src_dir in flir_splits.items():
        hr_dir = out_root / split / "HR"
        lr_dir = out_root / split / "LR"
        hr_dir.mkdir(parents=True, exist_ok=True)
        lr_dir.mkdir(parents=True, exist_ok=True)

        files = sorted(src_dir.iterdir())
        for f in files:
            img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise RuntimeError(f"unreadable image: {f}")
            hr = cv2.resize(img, HR_SIZE, interpolation=cv2.INTER_AREA)
            lr = cv2.resize(hr, LR_SIZE, interpolation=cv2.INTER_AREA)
            name = f.stem + ".png"
            cv2.imwrite(str(hr_dir / name), hr)
            cv2.imwrite(str(lr_dir / name), lr)
        print(f"{split}: {len(files)} pairs -> {out_root / split}")


def prepare_mlx_demo(out_dir: Path, thesis_repo: Path) -> None:
    mlx_dir = thesis_repo / "data" / "train" / "Original Data(32by24)"
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx in MLX_DEMO_FRAMES:
        src = mlx_dir / f"thermal_{idx}.png"
        img = cv2.imread(str(src), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError(f"unreadable image: {src}")
        if img.shape != (24, 32):
            raise RuntimeError(f"unexpected shape {img.shape} for {src}")
        cv2.imwrite(str(out_dir / src.name), img)
    print(f"demo: {len(MLX_DEMO_FRAMES)} MLX90640 frames -> {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, default=REPO_ROOT / "data",
        help="output root (default: <repo>/data)",
    )
    parser.add_argument(
        "--thesis-repo", type=Path, default=Path(DEFAULT_THESIS_REPO),
        help="checkout of sisr-embedded-systems holding the raw datasets",
    )
    args = parser.parse_args()

    prepare_flir(args.out / "flir_thermal_x2", args.thesis_repo)
    prepare_mlx_demo(args.out / "mlx90640_demo", args.thesis_repo)


if __name__ == "__main__":
    main()
