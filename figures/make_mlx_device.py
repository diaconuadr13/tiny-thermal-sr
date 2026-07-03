"""MLX90640 qualitative figure with REAL inference — including on-device rows.

Row 1 shows the firmware's embedded test frame (thermal_500.png); its SR
outputs are the int8 bytes captured from the boards' serial dumps
(measure/logs/<model>_<board>_sr.npy) — EDSR-Tiny from the Pico (does not
fit the ESP32), all other models from the ESP32. Rows 2-3 run the same
flashed flatbuffers on host with the LiteRT interpreter (device-identical
within <=4 LSB, see paper Sec. V-C). No ground truth exists at 64x48 for
MLX90640 data, so the figure is qualitative only.

Usage:
    python figures/make_mlx_device.py
Output:
    figures/mlx_device_qualitative.png / .pdf
"""
from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ai_edge_litert.interpreter import Interpreter

CAS_ROOT = Path(__file__).resolve().parents[1]
MLX = CAS_ROOT / "data" / "mlx90640_demo"
ART = CAS_ROOT / "export" / "artifacts"
LOGS = CAS_ROOT / "measure" / "logs"
OUT = CAS_ROOT / "figures"

MODELS = ["espcn_micro", "espcn_light", "fsrcnn_ds", "espcn", "edsr_tiny"]
PRETTY = {"espcn_micro": "ESPCN-Micro", "espcn_light": "ESPCN-Light",
          "fsrcnn_ds": "FSRCNN-ds", "espcn": "ESPCN", "edsr_tiny": "EDSR-Tiny"}
DEVICE_FRAME = "thermal_500.png"   # frame embedded in every firmware image
DEVICE_BOARD = {m: "esp32" for m in MODELS} | {"edsr_tiny": "pico"}


def load_interp(m: str) -> Interpreter:
    it = Interpreter(model_path=str(ART / m / f"{m}_int8.tflite"))
    it.allocate_tensors()
    return it


def int8_infer(interp: Interpreter, lr01: np.ndarray) -> np.ndarray:
    ind, outd = interp.get_input_details()[0], interp.get_output_details()[0]
    s_i, z_i = ind["quantization"]
    q = np.clip(np.round(lr01 / s_i + z_i), -128, 127).astype(np.int8)
    interp.set_tensor(ind["index"], q[None, :, :, None])
    interp.invoke()
    y = interp.get_tensor(outd["index"])[0, :, :, 0].astype(np.float32)
    s_o, z_o = outd["quantization"]
    return np.clip((y - z_o) * s_o, 0.0, 1.0)


def dequant_device_dump(interp: Interpreter, m: str) -> np.ndarray:
    """Dequantize the board-captured int8 SR dump with the model's output params."""
    raw = np.load(LOGS / f"{m}_{DEVICE_BOARD[m]}_sr.npy").astype(np.float32)
    s_o, z_o = interp.get_output_details()[0]["quantization"]
    return np.clip((raw - z_o) * s_o, 0.0, 1.0)


def main() -> None:
    interps = {m: load_interp(m) for m in MODELS}

    # rows: device frame first, then the two most structured remaining frames
    others = [p.name for p in sorted(MLX.glob("*.png")) if p.name != DEVICE_FRAME]
    others.sort(key=lambda n: -float(
        cv2.Laplacian(cv2.imread(str(MLX / n), 0).astype(np.float32), cv2.CV_32F).var()))
    frames = [DEVICE_FRAME] + others[:1]

    cols = ["LR $32{\\times}24$", "Bicubic"] + [PRETTY[m] for m in MODELS]
    fig, axes = plt.subplots(len(frames), len(cols), figsize=(7.16, 1.85), dpi=300)
    plt.rcParams.update({"font.size": 6, "font.family": "serif"})

    for r, name in enumerate(frames):
        lr_u8 = cv2.imread(str(MLX / name), 0)
        lr01 = lr_u8.astype(np.float32) / 255
        bi01 = np.clip(cv2.resize(lr_u8, (64, 48),
                                  interpolation=cv2.INTER_CUBIC).astype(np.float32) / 255, 0, 1)
        cells = [lr01, bi01]
        for m in MODELS:
            if r == 0:
                cells.append(dequant_device_dump(interps[m], m))
            else:
                cells.append(int8_infer(interps[m], lr01))

        for c, (img, title) in enumerate(zip(cells, cols)):
            ax = axes[r, c]
            up = cv2.resize(img, (64 * 4, 48 * 4), interpolation=cv2.INTER_NEAREST)
            ax.imshow(up, cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_linewidth(0.4)
            if r == 0:
                ax.set_title(title, fontsize=6, pad=2)
            if c == 0:
                ax.set_ylabel("on device" if r == 0 else "host (LiteRT)",
                              fontsize=5.5, labelpad=2)

    fig.subplots_adjust(wspace=0.04, hspace=0.06, left=0.025, right=0.995,
                        top=0.93, bottom=0.01)
    fig.savefig(OUT / "mlx_device_qualitative.pdf", bbox_inches="tight")
    fig.savefig(OUT / "mlx_device_qualitative.png", bbox_inches="tight")
    print(f"frames: {frames} -> {OUT}/mlx_device_qualitative.[pdf|png]")


if __name__ == "__main__":
    main()
