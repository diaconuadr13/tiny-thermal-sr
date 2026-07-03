"""Export the trained Keras models to full-integer INT8 TFLite + C headers.

Per model:
  1. Rebuild graph, load best weights (EDSR-Tiny: fold res_scale first).
  2. Convert with TFLiteConverter: TFLITE_BUILTINS_INT8, int8 input/output,
     representative dataset = training LR frames.
  3. Verify: int8 I/O dtypes AND the op set is exactly within the allowed
     TFLM vocabulary (CONV_2D, DEPTH_TO_SPACE, ADD, QUANTIZE) — any float op
     or hybrid kernel fails the export.
  4. Evaluate float32 vs INT8 PSNR/SSIM on the val split (quantization drop).
  5. Emit model C header + a firmware test-image header holding one MLX90640
     frame plus the PC-computed INT8 output as on-device reference.

Usage:
    python export/export_int8.py            # all models
    python export/export_int8.py espcn_micro
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import models_tf  # noqa: E402

DATA_ROOT = REPO_ROOT / "data" / "flir_thermal_x2"
DEMO_DIR = REPO_ROOT / "data" / "mlx90640_demo"
RUNS_ROOT = REPO_ROOT / "runs"
OUT_ROOT = REPO_ROOT / "export" / "artifacts"

LR_SIZE = (24, 32)  # (H, W)
SCALE = 2
TEST_FRAME = "thermal_500.png"
ALLOWED_OPS = {"CONV_2D", "DEPTH_TO_SPACE", "ADD", "QUANTIZE", "DEQUANTIZE"}


def load_trained(name: str) -> tf.keras.Model:
    weights = RUNS_ROOT / name / "best.weights.h5"
    if not weights.exists():
        raise FileNotFoundError(f"no trained weights: {weights}")
    model = models_tf.get_model(name, scale=SCALE, lr_size=LR_SIZE)
    model.load_weights(weights)
    if name == "edsr_tiny":
        model = models_tf.fold_edsr_res_scale(model, scale=SCALE,
                                              lr_size=LR_SIZE)
    return model


def representative_dataset():
    paths = sorted((DATA_ROOT / "train" / "LR").glob("*.png"))[:200]
    for p in paths:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        yield [(img.astype(np.float32) / 255.0)[None, :, :, None]]


def convert_int8(model: tf.keras.Model) -> bytes:
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = representative_dataset
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    return conv.convert()


def verify_full_int8(tfl: bytes, name: str) -> list[str]:
    interp = tf.lite.Interpreter(model_content=tfl)
    in_dt = interp.get_input_details()[0]["dtype"].__name__
    out_dt = interp.get_output_details()[0]["dtype"].__name__
    if in_dt != "int8" or out_dt != "int8":
        raise SystemExit(f"[{name}] NOT full-integer: in={in_dt} out={out_dt}")
    ops = sorted({op["op_name"] for op in interp._get_ops_details()})
    illegal = set(ops) - ALLOWED_OPS
    if illegal:
        raise SystemExit(f"[{name}] illegal ops for TFLM int8: {illegal}")
    return ops


def _int8_infer(interp: tf.lite.Interpreter, lr01: np.ndarray) -> np.ndarray:
    """Run one LR frame (float 0..1, HxW) through the int8 interpreter,
    return float 0..1 SR output."""
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    s_in, z_in = inp["quantization"]
    s_out, z_out = out["quantization"]
    q = np.clip(np.round(lr01 / s_in + z_in), -128, 127).astype(np.int8)
    interp.set_tensor(inp["index"], q[None, :, :, None])
    interp.invoke()
    y = interp.get_tensor(out["index"]).astype(np.float32)
    return (y[0, :, :, 0] - z_out) * s_out


def eval_val(model: tf.keras.Model, tfl: bytes) -> dict:
    interp = tf.lite.Interpreter(model_content=tfl)
    interp.allocate_tensors()
    names = sorted((DATA_ROOT / "val" / "LR").glob("*.png"))
    pf, sf, pq, sq = [], [], [], []
    for p in names:
        lr = cv2.imread(str(p), 0).astype(np.float32) / 255.0
        hr = cv2.imread(str(DATA_ROOT / "val" / "HR" / p.name), 0)
        hr = hr.astype(np.float32) / 255.0
        yf = np.clip(model(lr[None, :, :, None]).numpy()[0, :, :, 0], 0, 1)
        yq = np.clip(_int8_infer(interp, lr), 0, 1)
        hr_t, yf_t, yq_t = (x[..., None] for x in (hr, yf, yq))
        pf.append(float(tf.image.psnr(hr_t, yf_t, 1.0)))
        pq.append(float(tf.image.psnr(hr_t, yq_t, 1.0)))
        sf.append(float(tf.image.ssim(tf.constant(hr_t), tf.constant(yf_t), 1.0)))
        sq.append(float(tf.image.ssim(tf.constant(hr_t), tf.constant(yq_t), 1.0)))
    return {
        "float32": {"psnr": float(np.mean(pf)), "ssim": float(np.mean(sf))},
        "int8": {"psnr": float(np.mean(pq)), "ssim": float(np.mean(sq))},
        "psnr_drop": float(np.mean(pf) - np.mean(pq)),
    }


def c_array(data: bytes, var: str) -> str:
    lines = ["#pragma once", "#include <cstddef>", "#include <cstdint>", "",
             f"alignas(16) const unsigned char {var}[] = {{"]
    for i in range(0, len(data), 16):
        lines.append("  " + ", ".join(f"0x{b:02x}" for b in data[i:i + 16]) + ",")
    lines += ["};", f"const size_t {var}_len = {len(data)};", ""]
    return "\n".join(lines)


def int8_c_array(arr: np.ndarray, var: str, comment: str) -> str:
    flat = arr.flatten()
    lines = [f"// {comment}",
             f"const int8_t {var}[{flat.size}] = {{"]
    for i in range(0, flat.size, 16):
        lines.append("  " + ", ".join(str(int(v)) for v in flat[i:i + 16]) + ",")
    lines += ["};", ""]
    return "\n".join(lines)


def make_test_image_header(tfl: bytes, name: str, out_dir: Path) -> None:
    """Embed the MLX90640 test frame (quantized input) and the PC-computed
    INT8 output so firmware can check bit-exactness against the PC."""
    interp = tf.lite.Interpreter(model_content=tfl)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    s_in, z_in = inp["quantization"]

    lr = cv2.imread(str(DEMO_DIR / TEST_FRAME), 0).astype(np.float32) / 255.0
    q_in = np.clip(np.round(lr / s_in + z_in), -128, 127).astype(np.int8)
    interp.set_tensor(inp["index"], q_in[None, :, :, None])
    interp.invoke()
    q_out = interp.get_tensor(out["index"])[0, :, :, 0]

    h, w = lr.shape
    header = "\n".join([
        "#pragma once", "#include <cstdint>", "",
        f"// MLX90640 frame {TEST_FRAME}, quantized with the model's own",
        f"// input scale/zero-point (scale={s_in:.8f}, zp={z_in}).",
        f"const int kTestInputHeight = {h};",
        f"const int kTestInputWidth = {w};",
        f"const int kTestOutputHeight = {h * SCALE};",
        f"const int kTestOutputWidth = {w * SCALE};", "",
        int8_c_array(q_in, "test_input_int8", "quantized LR input (HxW)"),
        int8_c_array(q_out, "reference_output_int8",
                     "PC TFLite interpreter INT8 output (reference for "
                     "bit-exactness check)"),
    ])
    (out_dir / "test_image.h").write_text(header)


def export_one(name: str) -> dict:
    out_dir = OUT_ROOT / name
    out_dir.mkdir(parents=True, exist_ok=True)
    model = load_trained(name)

    tfl = convert_int8(model)
    ops = verify_full_int8(tfl, name)
    (out_dir / f"{name}_int8.tflite").write_bytes(tfl)
    (out_dir / f"{name}_int8_data.h").write_text(
        c_array(tfl, f"{name}_int8_tflite"))
    make_test_image_header(tfl, name, out_dir)

    metrics = eval_val(model, tfl)
    info = {
        "model": name,
        "params": int(model.count_params()),
        "tflite_int8_kb": round(len(tfl) / 1024, 1),
        "ops": ops,
        **metrics,
    }
    (out_dir / "export_info.json").write_text(json.dumps(info, indent=2))
    print(f"[{name}] {info['tflite_int8_kb']} KB ops={ops} "
          f"float PSNR={metrics['float32']['psnr']:.2f} "
          f"int8 PSNR={metrics['int8']['psnr']:.2f} "
          f"(drop {metrics['psnr_drop']:.3f} dB)")
    return info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="*",
                        default=list(models_tf.MODEL_BUILDERS))
    args = parser.parse_args()
    infos = [export_one(m) for m in args.models]
    (OUT_ROOT / "export_summary.json").write_text(json.dumps(infos, indent=2))


if __name__ == "__main__":
    main()
