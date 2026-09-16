"""Safe, idempotent aggregator for results/quantum_full/{dataset}_seed{seed}.json.

Run this any time - before, during (it'll just pick up whatever runs have
finished so far), or after the full grid completes. It never writes into the
per-run files, only rebuilds two summary files from scratch each time it's
called, so it's safe to run concurrently with run_full_grid_gpu.py (including
two parallel instances on two GPUs, see run_two_gpu.sh) with no locking needed.

Usage:
    python aggregate_results.py

Produces:
    results/quantum_full/_summary.json  - list of rows (one per dataset x seed,
                                           without the big objective_func_vals
                                           history, for quick inspection)
    results/quantum_full/_summary.csv   - same, as CSV for a quick look in
                                           Excel/pandas
    (also prints a mean +/- SD table per dataset to stdout, matching the
    format needed for the CMES Results section / comparison with the
    classical baselines in results/classical_grid_multiseed.json)
"""
import csv
import glob
import json
import os

import numpy as np

OUT_DIR = "results/quantum_full"


def main():
    paths = sorted(glob.glob(os.path.join(OUT_DIR, "*_seed*.json")))
    paths = [p for p in paths if not os.path.basename(p).startswith("_summary")]

    if not paths:
        print(f"No per-run result files found yet in {OUT_DIR}/. "
              f"Run run_full_grid_gpu.py (or run_two_gpu.sh) first.")
        return

    rows = []
    for p in paths:
        with open(p) as f:
            row = json.load(f)
        rows.append({k: v for k, v in row.items() if k != "objective_func_vals"})

    rows.sort(key=lambda r: (r["dataset"], r["seed"]))

    summary_json = os.path.join(OUT_DIR, "_summary.json")
    with open(summary_json, "w") as f:
        json.dump(rows, f, indent=2)

    summary_csv = os.path.join(OUT_DIR, "_summary.csv")
    fieldnames = list(rows[0].keys())
    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Aggregated {len(rows)} run(s) from {len(paths)} file(s) -> "
          f"{summary_json}, {summary_csv}\n")

    # Per-dataset mean +/- SD, the numbers that go straight into the paper.
    datasets = sorted(set(r["dataset"] for r in rows))
    print(f"{'dataset':10s} {'n_seeds':8s} {'valid_R2':>16s} {'valid_MAE':>16s} {'valid_RMSE':>16s}")
    for ds in datasets:
        sub = [r for r in rows if r["dataset"] == ds]
        r2 = np.array([r["valid_r2"] for r in sub])
        mae = np.array([r["valid_mae"] for r in sub])
        rmse = np.array([r["valid_rmse"] for r in sub])
        print(f"{ds:10s} {len(sub):<8d} "
              f"{r2.mean():+.3f} +/- {r2.std():.3f}   "
              f"{mae.mean():.4f} +/- {mae.std():.4f}   "
              f"{rmse.mean():.4f} +/- {rmse.std():.4f}")

    expected_total = None  # informational only; filled in by caller if desired
    n_expected = 4 * 5  # 4 datasets x 5 default seeds
    if len(rows) < n_expected:
        print(f"\n[note] {len(rows)}/{n_expected} expected runs present "
              f"(4 datasets x 5 seeds) - grid still in progress or run with "
              f"a --dataset/--seeds subset.")


if __name__ == "__main__":
    main()
