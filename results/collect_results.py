"""Aggregate the paper's main results table.

Merges:
  - export/artifacts/export_summary.json  (params, flash KB, float/int8 PSNR)
  - results/board_metrics.csv             (per-board latency, arena, status)
  - analytic MACs (conv: Hout*Wout*Cout*Kh*Kw*Cin; depth_to_space is free)
  - energy per inference = board active power x measured latency (ESTIMATED
    from datasheet currents, not measured - printed and stored as such)

Datasheet defaults (override via CLI):
  --esp32-ma  : ESP32 dual-core active current, RF off (datasheet, chip only)
  --pico-ma   : RP2040 active current (datasheet, chip only)
Both exclude board regulators/USB bridge; the paper labels the column
"estimated, chip-level".

Output: results/main_table.csv (+ pretty print)

Usage:
    python results/collect_results.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

EXPORT_SUMMARY = REPO_ROOT / "export" / "artifacts" / "export_summary.json"
BOARD_CSV = REPO_ROOT / "results" / "board_metrics.csv"
OUT_CSV = REPO_ROOT / "results" / "main_table.csv"

VOLTAGE = 3.3


def model_macs(name: str, lr_hw=(24, 32), scale: int = 2) -> int:
    """Analytic MAC count on the deployed graph (LR-space convs only)."""
    import models_tf
    model = models_tf.get_model(name, scale=scale, lr_size=lr_hw)
    macs = 0
    for layer in model.layers:
        cfg = layer.__class__.__name__
        if cfg != "Conv2D":
            continue
        k_h, k_w = layer.kernel_size
        c_in = layer.input.shape[-1]
        c_out = layer.filters
        h, w = layer.output.shape[1:3]
        macs += h * w * c_out * k_h * k_w * c_in
    return macs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--esp32-ma", type=float, default=50.0,
                        help="ESP32 active current mA (datasheet, RF off)")
    parser.add_argument("--pico-ma", type=float, default=25.0,
                        help="RP2040 active current mA (datasheet)")
    args = parser.parse_args()

    export = {e["model"]: e
              for e in json.loads(EXPORT_SUMMARY.read_text())}

    board: dict[tuple[str, str], dict] = {}
    if BOARD_CSV.exists():
        with BOARD_CSV.open() as f:
            for row in csv.DictReader(f):
                board[(row["model"], row["board"])] = row

    # kmacs is analytic and machine-independent; on boxes without TF/keras
    # (e.g. the flashing machine) reuse the values already in main_table.csv.
    prev_kmacs: dict[str, int] = {}
    if OUT_CSV.exists():
        with OUT_CSV.open() as f:
            for row in csv.DictReader(f):
                if row.get("kmacs", "").isdigit():
                    prev_kmacs[row["model"]] = int(row["kmacs"])

    current = {"esp32": args.esp32_ma, "pico": args.pico_ma}
    rows = []
    for name, e in export.items():
        try:
            macs = model_macs(name)
        except ImportError:
            if name not in prev_kmacs:
                raise SystemExit(
                    f"no TF/keras to compute MACs and no cached kmacs for {name}")
            macs = prev_kmacs[name] * 1e3
        row = {
            "model": name,
            "params": e["params"],
            "kmacs": round(macs / 1e3),
            "flash_kb": e["tflite_int8_kb"],
            "psnr_f32": round(e["float32"]["psnr"], 2),
            "ssim_f32": round(e["float32"]["ssim"], 4),
            "psnr_int8": round(e["int8"]["psnr"], 2),
            "psnr_drop": round(e["psnr_drop"], 3),
        }
        for b in ("esp32", "pico"):
            m = board.get((name, b))
            if m and m["status"] == "OK":
                lat = float(m["invoke_ms_mean"])
                row[f"{b}_ms"] = lat
                row[f"{b}_arena_kb"] = m["arena_used_kb"]
                row[f"{b}_mj_est"] = round(
                    VOLTAGE * current[b] * lat / 1000, 2)
                row[f"{b}_exact"] = "yes" if m["ref_mismatches"] == "0" else "NO"
            elif m:
                row[f"{b}_ms"] = m["status"]  # e.g. NOFIT_TENSORS
                row[f"{b}_arena_kb"] = row[f"{b}_mj_est"] = row[f"{b}_exact"] = "-"
            else:
                row[f"{b}_ms"] = "pending"
                row[f"{b}_arena_kb"] = row[f"{b}_mj_est"] = row[f"{b}_exact"] = "-"
        rows.append(row)

    rows.sort(key=lambda r: r["params"])
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"-> {OUT_CSV}   (energy: ESTIMATED, chip-level datasheet currents "
          f"esp32={args.esp32_ma}mA pico={args.pico_ma}mA @ {VOLTAGE}V)")
    for r in rows:
        print("  " + " | ".join(f"{k}={v}" for k, v in r.items()))


if __name__ == "__main__":
    main()
