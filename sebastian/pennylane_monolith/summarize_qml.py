"""Turns results_qml/monolith_qml.json into a readable table + CSV, mirroring
summarize_b.py from project_b_classical (same output shape, so the two are
easy to compare side by side).

Usage:
    python3 summarize_qml.py
"""
import csv
import json
import os

import numpy as np

IN_PATH = "results_qml/monolith_qml.json"
OUT_CSV_ALL = "results_qml/monolith_qml.csv"
OUT_CSV_SUMMARY = "results_qml/monolith_qml_summary.csv"
OUT_TXT_SUMMARY = "results_qml/monolith_qml_summary.txt"


def main():
    if not os.path.exists(IN_PATH):
        print(f"No results yet at {IN_PATH} - run sweep_monolith_qae.py first.")
        return
    rows = json.load(open(IN_PATH))
    if not rows:
        print(f"{IN_PATH} exists but is empty.")
        return

    rows.sort(key=lambda r: (r["tau"], r["sigma"], r["seed"]))
    with open(OUT_CSV_ALL, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    groups = {}
    for r in rows:
        key = (r["tau"], r["sigma"])
        groups.setdefault(key, []).append(r)

    summary_rows = []
    for (tau, sigma), sub in sorted(groups.items()):
        delta = np.array([r["delta_test_pct"] for r in sub])
        summary_rows.append({
            "tau": tau, "sigma": sigma, "n_seeds": len(sub),
            "n_weights": sub[0]["n_weights"],
            "delta_test_pct_mean": round(float(delta.mean()), 2),
            "delta_test_pct_std": round(float(delta.std()), 2),
        })

    with open(OUT_CSV_SUMMARY, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    lines = [f"{'tau':>5s} {'sigma':>6s} {'n':>3s} {'n_weights':>10s} {'Delta_test%':>16s}"]
    lines.append("-" * len(lines[0]))
    for s in summary_rows:
        lines.append(f"{s['tau']:>5d} {s['sigma']:>6.1f} {s['n_seeds']:>3d} "
                      f"{s['n_weights']:>10d} "
                      f"{s['delta_test_pct_mean']:+7.2f} +/- {s['delta_test_pct_std']:.2f}")
    text = "\n".join(lines)
    with open(OUT_TXT_SUMMARY, "w") as f:
        f.write(text + "\n")
    print(text)
    print(f"\nWrote {len(rows)} run(s) -> {OUT_CSV_ALL}")
    print(f"Wrote {len(summary_rows)} summary group(s) -> {OUT_CSV_SUMMARY}, {OUT_TXT_SUMMARY}")


if __name__ == "__main__":
    main()
