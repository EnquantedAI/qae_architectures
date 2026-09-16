"""
Regenerate the paper's parameter-matched benchmark figure from the aggregated
results, so the figure can never drift out of step with the table beside it.

    python3 aggregate_comparison.py          # writes results_tables/comparison_rows.csv
    python3 plot_matched_benchmark.py        # writes the figure next to it

Design notes (they are requirements, not taste):
  * One panel per series, shared y-axis - a single panel would need two scales
    for a range running from -76 to +86.
  * The two models differ by colour AND by line style AND by marker, so the
    figure survives greyscale printing and colour-vision deficiency.
  * A zero rule is drawn wherever a panel's data reaches it: on `energy` the
    QAE crosses it, and that crossing is the point of the figure.
  * Error bars are the same population SD (ddof=0) reported in the table.
"""
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)  # sebastian/ -- this script lives in sebastian/analysis/
ROWS = os.path.join(PARENT, "results_tables", "comparison_rows.csv")
OUT = os.path.join(PARENT, "figures",
                   "matched_pennylane_classical_benchmark.png")

QAE, CLS = "QAE", "TinyMLP (tuned)"
C_QAE, C_CLS = "#2a78d6", "#eb6834"          # validated categorical slots 1 and 2
INK, MUTED, GRID = "#1a1a19", "#52514e", "#d9d9d6"

PANELS = [
    ("mackey_glass_tau17", "Mackey–Glass  τ = 17"),
    ("mackey_glass_tau30", "Mackey–Glass  τ = 30"),
    ("energy", "Energy  (VIC demand)"),
    ("finance", "Finance  (AAPL close)"),
]


def load():
    if not os.path.exists(ROWS):
        sys.exit(f"missing {ROWS} - run aggregate_comparison.py first")
    data = {}
    for r in csv.DictReader(open(ROWS)):
        if r["model"] in (QAE, CLS):
            data[(r["dataset"], r["model"])] = data.get((r["dataset"], r["model"]), [])
            data[(r["dataset"], r["model"])].append(
                (float(r["sigma"]), float(r["delta_test_mean"]), float(r["delta_test_sd"])))
    for k in data:
        data[k].sort()
    return data


def main():
    data = load()
    missing = [f"{d}/{m}" for d, _ in PANELS for m in (QAE, CLS) if (d, m) not in data]
    if missing:
        print("warning: no rows for " + ", ".join(missing), file=sys.stderr)

    fig, axes = plt.subplots(1, 4, figsize=(11.6, 3.5), sharey=True)
    fig.patch.set_facecolor("white")

    for ax, (ds, title) in zip(axes, PANELS):
        ax.set_facecolor("white")
        for model, colour, style, marker in (
                (CLS, C_CLS, "-", "o"),
                (QAE, C_QAE, "--", "s")):
            pts = data.get((ds, model))
            if not pts:
                continue
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            es = [p[2] for p in pts]
            ax.errorbar(xs, ys, yerr=es, color=colour, linestyle=style, marker=marker,
                        markersize=5.5, linewidth=1.8, capsize=3, elinewidth=1.1,
                        markeredgecolor="white", markeredgewidth=0.9, zorder=3)

        lo, hi = ax.get_ylim()
        if lo < 0 < hi:
            ax.axhline(0, color=MUTED, linewidth=0.9, zorder=2)

        ax.set_title(title, fontsize=9.5, color=INK, pad=8)
        ax.set_xlabel("noise level  σ", fontsize=9, color=MUTED)
        ax.set_xticks([0.1, 0.2, 0.3])
        ax.tick_params(labelsize=8.5, colors=MUTED, length=3)
        ax.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)

    axes[0].set_ylabel(r"$\Delta_{test}$  (%)", fontsize=9.5, color=INK)

    fig.legend(handles=[
        Line2D([], [], color=C_CLS, linestyle="-", marker="o", markersize=5.5,
               markeredgecolor="white", linewidth=1.8, label="TinyMLP, 142 parameters"),
        Line2D([], [], color=C_QAE, linestyle="--", marker="s", markersize=5.5,
               markeredgecolor="white", linewidth=1.8, label="Monolith QAE, 144 parameters"),
    ], loc="lower center", ncol=2, frameon=False, fontsize=9.5,
        bbox_to_anchor=(0.5, -0.02))

    fig.tight_layout(rect=(0, 0.08, 1, 1))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=200, facecolor="white")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
