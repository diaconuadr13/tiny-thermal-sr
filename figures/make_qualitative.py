"""Qualitative MLX90640 demo figure: LR / bicubic / each INT8 model SR.

Runs every exported INT8 .tflite on the native 32x24 MLX90640 demo frames
with the PC TFLite interpreter (the same artifact that runs on the boards)
and assembles a grid: rows = frames, columns = LR, bicubic x2, models.

No PSNR is computed - these frames have no ground truth (native sensor
resolution). Qualitative only.

Usage:
    python figures/make_qualitative.py
Outputs:
    figures/qualitative_grid.png       (all demo frames)
    figures/qualitative_paper.png      (3 frames selected for the paper)
    figures/tiles/<frame>_<column>.png (individual tiles for LaTeX)
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

DEMO_DIR = REPO_ROOT / "data" / "mlx90640_demo"
ARTIFACTS = REPO_ROOT / "export" / "artifacts"
OUT_DIR = REPO_ROOT / "figures"
TILES = OUT_DIR / "tiles"

MODEL_ORDER = ["espcn_micro", "espcn_light", "espcn", "fsrcnn_ds", "edsr_tiny"]
PAPER_FRAMES = ["thermal_500.png", "thermal_900.png", "thermal_1300.png"]
ZOOM = 4  # display upscale (nearest) so 64x48 tiles are visible in print


def int8_sr(tfl_path: Path, lr01: np.ndarray) -> np.ndarray:
    interp = tf.lite.Interpreter(model_path=str(tfl_path))
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    s_in, z_in = inp["quantization"]
    s_out, z_out = out["quantization"]
    q = np.clip(np.round(lr01 / s_in + z_in), -128, 127).astype(np.int8)
    interp.set_tensor(inp["index"], q[None, :, :, None])
    interp.invoke()
    y = (interp.get_tensor(out["index"]).astype(np.float32) - z_out) * s_out
    return np.clip(y[0, :, :, 0], 0, 1)


def to_u8(img01: np.ndarray) -> np.ndarray:
    return (np.clip(img01, 0, 1) * 255).round().astype(np.uint8)


def upscale(img: np.ndarray, factor: int) -> np.ndarray:
    return cv2.resize(img, None, fx=factor, fy=factor,
                      interpolation=cv2.INTER_NEAREST)


def main() -> None:
    TILES.mkdir(parents=True, exist_ok=True)
    frames = sorted(DEMO_DIR.glob("*.png"))
    columns = ["lr", "bicubic"] + MODEL_ORDER

    grid_rows = []
    for f in frames:
        lr = cv2.imread(str(f), 0).astype(np.float32) / 255.0
        tiles = {
            "lr": upscale(to_u8(lr), 2 * ZOOM),  # 2x extra so LR matches SR size
            "bicubic": upscale(to_u8(np.clip(cv2.resize(
                lr, (64, 48), interpolation=cv2.INTER_CUBIC), 0, 1)), ZOOM),
        }
        for m in MODEL_ORDER:
            tfl = ARTIFACTS / m / f"{m}_int8.tflite"
            tiles[m] = upscale(to_u8(int8_sr(tfl, lr)), ZOOM)

        for col, img in tiles.items():
            cv2.imwrite(str(TILES / f"{f.stem}_{col}.png"), img)

        sep = np.full((tiles["lr"].shape[0], 4), 255, np.uint8)
        row = []
        for col in columns:
            row.extend([tiles[col], sep])
        grid_rows.append(np.hstack(row[:-1]))

    h_sep = np.full((4, grid_rows[0].shape[1]), 255, np.uint8)
    full = []
    for r in grid_rows:
        full.extend([r, h_sep])
    cv2.imwrite(str(OUT_DIR / "qualitative_grid.png"), np.vstack(full[:-1]))

    paper_rows = [grid_rows[i] for i, f in enumerate(frames)
                  if f.name in PAPER_FRAMES]
    paper = []
    for r in paper_rows:
        paper.extend([r, h_sep])
    cv2.imwrite(str(OUT_DIR / "qualitative_paper.png"), np.vstack(paper[:-1]))
    print(f"columns: {columns}")
    print(f"{len(frames)} frames -> qualitative_grid.png; "
          f"{len(paper_rows)} -> qualitative_paper.png; tiles/ per-cell")


if __name__ == "__main__":
    main()
