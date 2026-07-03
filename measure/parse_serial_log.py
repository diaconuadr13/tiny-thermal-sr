"""Parse tiny-thermal-sr firmware serial logs into results/board_metrics.csv.

Each log is one board run of one model, captured with capture_serial.py (or
any serial monitor) and saved as measure/logs/<model>_<board>.log.
Lines of interest all start with "CAS:".

Usage:
    python measure/parse_serial_log.py             # parse all logs
    python measure/parse_serial_log.py logs/espcn_micro_esp32.log
"""

from __future__ import annotations

import argparse
import csv
import re
import statistics
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = REPO_ROOT / "measure" / "logs"
OUT_CSV = REPO_ROOT / "results" / "board_metrics.csv"

FIELDS = [
    "model", "board", "status", "model_kb", "arena_alloc_kb",
    "arena_used_kb", "heap_before_kb", "heap_after_kb",
    "invoke_ms_first", "invoke_ms_mean", "invoke_ms_std", "samples",
    "ref_mismatches", "ref_max_abs_diff",
]


def parse_log(path: Path) -> dict:
    kv: dict[str, str] = {}
    samples: list[float] = []
    sr_dump: str | None = None
    for line in path.read_text(errors="replace").splitlines():
        m = re.search(r"CAS:([a-z0-9_]+)=(.*)", line.strip())
        if not m:
            continue
        key, val = m.group(1), m.group(2).strip()
        if key == "sample_us":
            samples.append(float(val) / 1000.0)
        elif key == "sr_int8":
            sr_dump = val
        else:
            kv[key] = val

    def kb(key):
        return round(float(kv[key]) / 1024, 1) if key in kv else ""

    row = {
        "model": kv.get("model", path.stem),
        "board": kv.get("board", ""),
        "status": kv.get("status", "INCOMPLETE"),
        "model_kb": kb("model_bytes"),
        "arena_alloc_kb": kb("arena_alloc"),
        "arena_used_kb": kb("arena_used"),
        "heap_before_kb": kb("heap_before"),
        "heap_after_kb": kb("heap_after"),
        "invoke_ms_first": (round(float(kv["invoke_us_first"]) / 1000, 2)
                            if "invoke_us_first" in kv else ""),
        "invoke_ms_mean": round(statistics.mean(samples), 2) if samples else "",
        "invoke_ms_std": (round(statistics.stdev(samples), 3)
                          if len(samples) > 1 else ""),
        "samples": len(samples),
        "ref_mismatches": kv.get("ref_mismatches", ""),
        "ref_max_abs_diff": kv.get("ref_max_abs_diff", ""),
    }

    if sr_dump:
        vals = np.array([int(v) for v in sr_dump.split(",")], dtype=np.int8)
        out = LOG_DIR / f"{row['model']}_{row['board']}_sr.npy"
        np.save(out, vals.reshape(48, 64))
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="*", type=Path,
                        default=sorted(LOG_DIR.glob("*.log")))
    args = parser.parse_args()
    if not args.logs:
        raise SystemExit(f"no logs found in {LOG_DIR}")

    rows = [parse_log(p) for p in args.logs]
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} runs -> {OUT_CSV}")
    for r in rows:
        print(f"  {r['model']:12s} {r['board']:5s} {r['status']:10s} "
              f"mean={r['invoke_ms_mean']}ms exact="
              f"{'yes' if r['ref_mismatches'] == '0' else r['ref_mismatches']}")


if __name__ == "__main__":
    main()
