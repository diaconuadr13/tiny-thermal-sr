"""PSNR-vs-latency trade-off scatter for the paper (print/grayscale-safe).

Reads results/main_table.csv (board latency columns filled by
collect_results.py after board runs). Each model appears once per board it
runs on; the two boards' points for the same model are joined by a thin gray
line so cross-board pairs are visually explicit. Boards are distinguished by
marker shape AND color (Okabe-Ito CVD-safe pair). Model names are direct
labels with hand-placed offsets (no collisions at IEEE column width); the
bicubic baseline is a dashed reference line.

Usage:
    python figures/make_scatter.py
Output:
    figures/psnr_vs_latency.pdf / .png
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CAS_ROOT = Path(__file__).resolve().parents[1]
TABLE = CAS_ROOT / "results" / "main_table.csv"
OUT = CAS_ROOT / "figures"

BICUBIC_DB = 30.56  # bicubic x2 PSNR on the val split (see paper Table II)

BOARDS = {
    "esp32": {"color": "#0072B2", "marker": "o", "label": "ESP32 (240 MHz)"},
    "pico": {"color": "#D55E00", "marker": "^", "label": "Pico / RP2040 (133 MHz)"},
}
PRETTY = {
    "espcn_micro": "ESPCN-Micro", "espcn_light": "ESPCN-Light",
    "espcn": "ESPCN", "fsrcnn_ds": "FSRCNN-ds", "edsr_tiny": "EDSR-Tiny",
}
# hand-placed label anchors: (x, y, ha, va) in data coords, tuned for
# figsize=(3.5, 2.2) log-x. Anchored per model, NOT per point.
LABELS = {
    "espcn_micro": (160, 30.87, "center", "bottom"),
    "espcn_light": (380, 30.68, "center", "top"),
    "fsrcnn_ds":   (670, 31.24, "center", "bottom"),
    "espcn":       (1210, 31.56, "center", "bottom"),
    "edsr_tiny":   (8200, 32.58, "right", "center"),
}


def main() -> None:
    with TABLE.open() as f:
        rows = list(csv.DictReader(f))

    plt.rcParams.update({
        "font.size": 8, "font.family": "serif",
        "axes.linewidth": 0.6, "grid.linewidth": 0.4,
    })
    fig, ax = plt.subplots(figsize=(3.5, 2.2), dpi=300)  # IEEE column width

    # per-model points on each board (skip NOFIT/pending)
    pts: dict[str, dict[str, tuple[float, float]]] = {}
    for r in rows:
        for board in BOARDS:
            try:
                lat = float(r[f"{board}_ms"])
            except ValueError:
                continue
            pts.setdefault(r["model"], {})[board] = (lat, float(r["psnr_int8"]))
    if not pts:
        raise SystemExit("no board latency data in main_table.csv yet")

    # bicubic reference
    ax.axhline(BICUBIC_DB, color="#999999", lw=0.7, ls=(0, (4, 3)), zorder=1)
    ax.text(9500, BICUBIC_DB + 0.03, f"bicubic {BICUBIC_DB:.2f} dB",
            fontsize=6.5, color="#777777", ha="right", va="bottom")

    # connect cross-board pairs, then draw points
    for model, bp in pts.items():
        if len(bp) == 2:
            (x1, y1), (x2, y2) = bp["esp32"], bp["pico"]
            ax.plot([x1, x2], [y1, y2], color="#BBBBBB", lw=0.7, zorder=2)
    for board, style in BOARDS.items():
        xs = [bp[board][0] for bp in pts.values() if board in bp]
        ys = [bp[board][1] for bp in pts.values() if board in bp]
        ax.scatter(xs, ys, s=26, c=style["color"], marker=style["marker"],
                   label=style["label"], zorder=3,
                   edgecolors="white", linewidths=0.5)

    # one hand-placed label per model
    for model, (x, y, ha, va) in LABELS.items():
        if model not in pts:
            continue
        name = PRETTY[model]
        if model == "edsr_tiny" and "esp32" not in pts[model]:
            name += "\n(Pico only)"
        ax.text(x, y, name, fontsize=6.5, color="#333333", ha=ha, va=va,
                linespacing=1.1, zorder=4)

    ax.set_xscale("log")
    ax.set_xlim(80, 16000)
    ax.set_ylim(30.4, 32.85)
    ax.set_xlabel("Inference latency (ms, log scale)")
    ax.set_ylabel("INT8 PSNR (dB)")
    ax.grid(True, which="both", alpha=0.2)
    ax.legend(frameon=True, framealpha=0.9, edgecolor="#CCCCCC",
              loc="upper left", fontsize=6.5, borderpad=0.5)
    fig.tight_layout()
    fig.savefig(OUT / "psnr_vs_latency.pdf")
    fig.savefig(OUT / "psnr_vs_latency.png")
    print(f"-> {OUT}/psnr_vs_latency.[pdf|png]")


if __name__ == "__main__":
    main()
