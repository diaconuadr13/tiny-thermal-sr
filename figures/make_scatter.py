"""PSNR-vs-latency trade-off scatter for the paper (print/grayscale-safe).

Reads results/main_table.csv (needs board latency columns filled by
collect_results.py after board runs). One axis pair; boards distinguished by
marker shape AND color (Okabe-Ito CVD-safe pair); every point direct-labeled
with its model name. Models that do not fit a board simply have no point.

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

REPO_ROOT = Path(__file__).resolve().parents[1]
TABLE = REPO_ROOT / "results" / "main_table.csv"
OUT = REPO_ROOT / "figures"

BOARDS = {
    "esp32": {"color": "#0072B2", "marker": "o", "label": "ESP32"},
    "pico": {"color": "#D55E00", "marker": "^", "label": "Pico (RP2040)"},
}
PRETTY = {
    "espcn_micro": "ESPCN-Micro", "espcn_light": "ESPCN-Light",
    "espcn": "ESPCN", "fsrcnn_ds": "FSRCNN-ds", "edsr_tiny": "EDSR-Tiny",
}


def main() -> None:
    with TABLE.open() as f:
        rows = list(csv.DictReader(f))

    plt.rcParams.update({
        "font.size": 8, "font.family": "serif",
        "axes.linewidth": 0.6, "grid.linewidth": 0.4,
    })
    fig, ax = plt.subplots(figsize=(3.5, 2.4), dpi=300)  # IEEE column width

    plotted = False
    for board, style in BOARDS.items():
        pts = []
        for r in rows:
            try:
                lat = float(r[f"{board}_ms"])
            except ValueError:
                continue  # NOFIT / pending
            pts.append((lat, float(r["psnr_int8"]), PRETTY[r["model"]]))
        if not pts:
            continue
        plotted = True
        xs, ys, names = zip(*pts)
        ax.scatter(xs, ys, s=22, c=style["color"], marker=style["marker"],
                   label=style["label"], zorder=3,
                   edgecolors="white", linewidths=0.5)
        for x, y, name in pts:
            ax.annotate(name, (x, y), textcoords="offset points",
                        xytext=(4, 4), fontsize=6.5, color="#333333")

    if not plotted:
        raise SystemExit("no board latency data in main_table.csv yet")

    ax.set_xscale("log")
    ax.set_xlabel("Inference latency (ms, log scale)")
    ax.set_ylabel("PSNR (dB), INT8, FLIR-thermal val")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False, loc="lower right", fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "psnr_vs_latency.pdf")
    fig.savefig(OUT / "psnr_vs_latency.png")
    print(f"-> {OUT}/psnr_vs_latency.[pdf|png]")


if __name__ == "__main__":
    main()
