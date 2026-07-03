"""FLIR val qualitative figure WITH ground truth: LR / bicubic / 5 INT8 models / GT.

Runs the deployed int8 .tflite models (export/artifacts/<m>/<m>_int8.tflite)
with the LiteRT interpreter on FLIR validation frames — the same flatbuffers
that were flashed to the boards — and annotates per-frame PSNR vs the 64x48
ground truth. Frame selection is rule-based (not cherry-picked): the frame at
the 25th percentile of bicubic PSNR (a harder scene) and the frame with the
highest Laplacian variance (the most spatial detail).

Usage:
    python figures/make_flir_gt.py
Output:
    figures/flir_gt_comparison.png / .pdf
"""
from __future__ import annotations

import math
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ai_edge_litert.interpreter import Interpreter

CAS_ROOT = Path(__file__).resolve().parents[1]
VAL = CAS_ROOT / "data" / "flir_thermal_x2" / "val"
ART = CAS_ROOT / "export" / "artifacts"
OUT = CAS_ROOT / "figures"

MODELS = ["espcn_micro", "espcn_light", "fsrcnn_ds", "espcn", "edsr_tiny"]
PRETTY = {"espcn_micro": "ESPCN-Micro", "espcn_light": "ESPCN-Light",
          "fsrcnn_ds": "FSRCNN-ds", "espcn": "ESPCN", "edsr_tiny": "EDSR-Tiny"}


def psnr01(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a - b) ** 2))
    return 10 * math.log10(1.0 / mse) if mse > 0 else float("inf")


def int8_infer(interp: Interpreter, lr01: np.ndarray) -> np.ndarray:
    """Full-int8 inference of one (24,32) [0,1] frame -> (48,64) [0,1]."""
    ind = interp.get_input_details()[0]
    outd = interp.get_output_details()[0]
    s_i, z_i = ind["quantization"]
    q = np.clip(np.round(lr01 / s_i + z_i), -128, 127).astype(np.int8)
    interp.set_tensor(ind["index"], q[None, :, :, None])
    interp.invoke()
    y = interp.get_tensor(outd["index"])[0, :, :, 0].astype(np.float32)
    s_o, z_o = outd["quantization"]
    return np.clip((y - z_o) * s_o, 0.0, 1.0)


def main() -> None:
    interps = {}
    for m in MODELS:
        it = Interpreter(model_path=str(ART / m / f"{m}_int8.tflite"))
        it.allocate_tensors()
        interps[m] = it

    names = sorted(p.name for p in (VAL / "HR").iterdir() if p.suffix == ".png")
    # rule-based picks: 25th-percentile bicubic PSNR + highest-detail frame
    by_psnr, by_detail = [], []
    for n in names:
        hr = cv2.imread(str(VAL / "HR" / n), 0).astype(np.float32) / 255
        lr = cv2.imread(str(VAL / "LR" / n), 0)
        bi = cv2.resize(lr, (64, 48), interpolation=cv2.INTER_CUBIC).astype(np.float32) / 255
        by_psnr.append((psnr01(hr, bi), n))
        by_detail.append((float(cv2.Laplacian(hr, cv2.CV_32F).var()), n))
    by_psnr.sort()
    by_detail.sort(reverse=True)
    picks = [by_psnr[len(by_psnr) // 4][1]]
    picks.append(next(n for _, n in by_detail if n not in picks))

    cols = ["LR $32{\\times}24$", "Bicubic"] + [PRETTY[m] for m in MODELS] + ["GT $64{\\times}48$"]
    fig, axes = plt.subplots(len(picks), len(cols), figsize=(7.16, 2.05), dpi=300)
    plt.rcParams.update({"font.size": 6, "font.family": "serif"})

    for r, n in enumerate(picks):
        hr01 = cv2.imread(str(VAL / "HR" / n), 0).astype(np.float32) / 255
        lr_u8 = cv2.imread(str(VAL / "LR" / n), 0)
        lr01 = lr_u8.astype(np.float32) / 255
        bi01 = np.clip(cv2.resize(lr_u8, (64, 48),
                                  interpolation=cv2.INTER_CUBIC).astype(np.float32) / 255, 0, 1)
        cells = [(lr01, None), (bi01, psnr01(hr01, bi01))]
        for m in MODELS:
            sr01 = int8_infer(interps[m], lr01)
            cells.append((sr01, psnr01(hr01, sr01)))
        cells.append((hr01, None))

        for c, ((img, p), title) in enumerate(zip(cells, cols)):
            ax = axes[r, c]
            up = cv2.resize(img, (64 * 4, 48 * 4), interpolation=cv2.INTER_NEAREST)
            ax.imshow(up, cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_linewidth(0.4)
            if r == 0:
                ax.set_title(title, fontsize=6, pad=2)
            if p is not None:
                ax.set_xlabel(f"{p:.2f} dB", fontsize=5.5, labelpad=1)

    fig.subplots_adjust(wspace=0.04, hspace=0.16, left=0.01, right=0.99,
                        top=0.90, bottom=0.06)
    fig.savefig(OUT / "flir_gt_comparison.pdf", bbox_inches="tight")
    fig.savefig(OUT / "flir_gt_comparison.png", bbox_inches="tight")
    print(f"frames: {picks} -> {OUT}/flir_gt_comparison.[pdf|png]")


if __name__ == "__main__":
    main()
