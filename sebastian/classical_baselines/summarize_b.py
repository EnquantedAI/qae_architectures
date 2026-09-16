"""Turns results_b/classical_grid_b.json into something readable: a CSV you
can open in Excel/Numbers, and a mean +/- SD table printed to the screen
(and saved to a .txt file), grouped by dataset x model x sigma - the numbers
that go straight into the paper's comparison table.

Safe to run any time - before, during (it just summarizes whatever rows
exist so far), or after the full grid finishes. Never modifies
classical_grid_b.json itself.

Usage:
    python3 summarize_b.py

Produces (in results_b/):
    classical_grid_b.csv     - every individual run, one row each (all seeds)
    classical_grid_summary.csv   - mean +/- SD per dataset x model x sigma
    classical_grid_summary.txt   - same, as a plain-text table (also printed)
"""
import csv
import json
import os

import numpy as np

IN_PATH = "results_b/classical_grid_b.json"
OUT_CSV_ALL = "results_b/classical_grid_b.csv"
OUT_CSV_SUMMARY = "results_b/classical_grid_summary.csv"
OUT_TXT_SUMMARY = "results_b/classical_grid_summary.txt"


def main():
    if not os.path.exists(IN_PATH):
        print(f"No results yet at {IN_PATH} - run run_classical_grid_b.py first.")
        return

    with open(IN_PATH) as f:
        rows = json.load(f)

    if not rows:
        print(f"{IN_PATH} exists but is empty - run run_classical_grid_b.py first.")
        return

    rows.sort(key=lambda r: (r["dataset"], r["model"], r["sigma"], r["seed"]))

    # --- 1. flat CSV of every individual run ---
    fieldnames = list(rows[0].keys())
    with open(OUT_CSV_ALL, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # --- 2. mean +/- SD summary, grouped by dataset x model x sigma ---
    groups = {}
    for r in rows:
        key = (r["dataset"], r["model"], r["sigma"])
        groups.setdefault(key, []).append(r)

    summary_rows = []
    for (ds, model, sigma), sub in sorted(groups.items()):
        r2 = np.array([r["test_r2"] for r in sub])
        rmse = np.array([r["test_rmse"] for r in sub])
        mae = np.array([r["test_mae"] for r in sub])
        delta = np.array([r["delta_test_pct"] for r in sub])
        summary_rows.append({
            "dataset": ds, "model": model, "sigma": sigma, "n_seeds": len(sub),
            "n_params": sub[0]["n_params"],
            "test_r2_mean": round(float(r2.mean()), 4),
            "test_r2_std": round(float(r2.std()), 4),
            "test_rmse_mean": round(float(rmse.mean()), 4),
            "test_mae_mean": round(float(mae.mean()), 4),
            "delta_test_pct_mean": round(float(delta.mean()), 2),
            "delta_test_pct_std": round(float(delta.std()), 2),
        })

    with open(OUT_CSV_SUMMARY, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    # --- 3. plain-text table (printed + saved) ---
    lines = []
    header = (f"{'dataset':18s} {'model':8s} {'sigma':>5s} {'n':>3s} {'params':>7s} "
              f"{'test_R2':>14s} {'Delta_test%':>16s}")
    lines.append(header)
    lines.append("-" * len(header))
    for s in summary_rows:
        lines.append(
            f"{s['dataset']:18s} {s['model']:8s} {s['sigma']:>5.1f} {s['n_seeds']:>3d} "
            f"{s['n_params']:>7d} "
            f"{s['test_r2_mean']:+.3f} +/- {s['test_r2_std']:.3f}  "
            f"{s['delta_test_pct_mean']:+7.2f} +/- {s['delta_test_pct_std']:.2f}"
        )
    text = "\n".join(lines)
    with open(OUT_TXT_SUMMARY, "w") as f:
        f.write(text + "\n")

    print(text)
    print(f"\nWrote {len(rows)} individual run(s) -> {OUT_CSV_ALL}")
    print(f"Wrote {len(summary_rows)} summary group(s) -> {OUT_CSV_SUMMARY}, {OUT_TXT_SUMMARY}")


if __name__ == "__main__":
    main()
