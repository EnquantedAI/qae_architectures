"""Summarize canonical QAE runs without pooling datasets or smoke tests.

Run from this directory: python summarize_qml.py
Raw JSON is preserved; Delta at zero noise is reported as undefined.
"""
import csv
import json
from pathlib import Path
import statistics as st

ROOT = Path(__file__).resolve().parent


def main():
    rows = json.loads((ROOT / "results_qml/monolith_qml.json").read_text())
    def dataset(row):
        name = row.get("dataset", "mackey_glass")
        return f"mackey_glass_tau{row['tau']}" if name == "mackey_glass" else name
    max_epochs = {}
    for row in rows:
        name = dataset(row)
        max_epochs[name] = max(max_epochs.get(name, 0), row["n_epochs"])
    rows = [row for row in rows if row["n_epochs"] == max_epochs[dataset(row)]]
    rows.sort(key=lambda row: (dataset(row), row["sigma"], row["seed"]))
    flat = [{**row, "dataset": dataset(row),
             "delta_test_pct": row["delta_test_pct"] if row["sigma"] > 0 else None}
            for row in rows]
    fields = list(dict.fromkeys(key for row in flat for key in row))
    with (ROOT / "results_qml/monolith_qml.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(flat)
    groups = {}
    for row in rows:
        groups.setdefault((dataset(row), row["sigma"]), []).append(row)
    summaries, lines = [], ["Canonical runs; population SD; Delta is undefined at sigma=0."]
    for (name, sigma), sub in sorted(groups.items()):
        values = [row["delta_test_pct"] for row in sub] if sigma > 0 else []
        summary = {"dataset": name, "sigma": sigma, "n_seeds": len(sub),
                   "n_weights": sub[0]["n_weights"],
                   "mse_test_recovered_mean": st.mean(row["mse_test_recovered"] for row in sub),
                   "delta_test_pct_mean": st.mean(values) if values else None,
                   "delta_test_pct_std": st.pstdev(values) if values else None}
        summaries.append(summary)
        text = f"{st.mean(values):+.2f} +/- {st.pstdev(values):.2f}" if values else "undefined"
        lines.append(f"{name:22s} sigma={sigma:.2f} n={len(sub)} Delta={text}")
    with (ROOT / "results_qml/monolith_qml_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(summaries)
    text = "\n".join(line.rstrip() for line in lines) + "\n"
    (ROOT / "results_qml/monolith_qml_summary.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
