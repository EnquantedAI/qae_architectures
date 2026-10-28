"""Verify recorded extension results without training or writing any files."""

import ast
import csv
import hashlib
import json
from pathlib import Path
import statistics as st
import sys

import numpy as np

import aggregate_comparison as comparison

ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = json.loads((ROOT / "results_tables/raw_result_manifest.json").read_text())
    for entry in manifest["files"]:
        path = ROOT / entry["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"], path
        assert len(json.loads(path.read_text())) == entry["rows"], path
    print("PASS: raw result hashes and row counts match the synchronization manifest.")

    quantum = json.loads(Path(comparison.QML_ROWS).read_text())
    classical = json.loads(Path(comparison.CLS_TUNED).read_text())
    selections = json.loads(Path(comparison.CLS_SELECTED).read_text())
    selected = {(r["dataset"], r["model"]): r["selected"]["config"] for r in selections}
    canonical = {}
    historical_zero = 0
    for name, rows, noise_key in (
        ("QAE", quantum, "mse_test_noise"),
        ("classical", classical, "mse_test_noisy"),
    ):
        for row in rows:
            if noise_key not in row:
                continue
            if row["sigma"] > 0:
                expected = 100 * (1 - row["mse_test_recovered"] / row[noise_key])
                assert abs(row["delta_test_pct"] - expected) < 1e-8, row
            else:
                assert row[noise_key] == 0, row
                historical_zero += row["delta_test_pct"] != 0
            assert row["mse_test_recovered"] >= 0, row
            if name == "QAE":
                if row["n_epochs"] != 300:
                    continue
                dataset = comparison._quantum_dataset_label(row)
                model = "QAE"
            else:
                if row.get("stage") != "final":
                    continue
                dataset, model = row["dataset"], row["model"]
                if row["config"] != selected[(dataset, model)]:
                    continue
                if row["seed"] not in comparison.CLS_FINAL_SEEDS:
                    continue
            key = (dataset, model, row["sigma"])
            canonical.setdefault(key, []).append(row)
    datasets = ("mackey_glass_tau17", "mackey_glass_tau30", "energy", "finance")
    sigmas = [0.0] + [round(0.1 + 0.05 * i, 2) for i in range(9)]
    for dataset in datasets:
        for model in ("QAE", "TinyMLP", "MLP-AE", "LSTM-AE"):
            for sigma in sigmas:
                rows = canonical[(dataset, model, sigma)]
                assert len(rows) == 7, (dataset, model, sigma, len(rows))
                assert {r["seed"] for r in rows} == set(range(2023, 2030))
    print("PASS: positive-noise Delta formulas and 160 canonical seven-seed cells.")
    print(f"INFO: {historical_zero} historical zero-noise Delta placeholders are ignored.")

    cells, _ = comparison.collect()
    csv_rows = list(csv.DictReader((ROOT / "results_tables/comparison_rows.csv").open()))
    for row in csv_rows:
        cell = cells[(row["dataset"], float(row["sigma"]))][row["model"]]
        assert int(row["n_seeds"]) == cell["n"]
        assert abs(float(row["delta_test_mean"]) - cell["mean"]) <= 0.000051
        assert abs(float(row["delta_test_sd"]) - cell["sd"]) <= 0.000051
    expected_rows = sum(len(v) for (dataset, _), v in cells.items() if dataset != "beer")
    assert len(csv_rows) == expected_rows
    print("PASS: stored comparison CSV agrees with canonical raw results.")

    sys.path.insert(0, str(ROOT / "classical_baselines"))
    sys.path.insert(0, str(ROOT / "pennylane_monolith"))
    import datasets_b
    import datasets_qml

    # Execute only the numerical generator/scaler bodies, avoiding framework imports.
    path = ROOT / "pennylane_monolith/train_monolith_qae.py"
    tree = ast.parse(path.read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name in ("mackey_glass", "x2y")]
    namespace = {"np": np}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)
    for dataset in datasets:
        _, _, classical_series = datasets_b.prepare_dataset(dataset)
        if dataset.startswith("mackey_glass"):
            raw = namespace["mackey_glass"](length=1200, beta=0.17, tau=int(dataset[-2:]))
            quantum_series = namespace["x2y"](raw, xlim=(raw.min(), raw.max()),
                                              ylim=(0, 1))[3::6]
        else:
            quantum_series = datasets_qml.load_realworld_series(dataset)
        np.testing.assert_allclose(classical_series, quantum_series, rtol=0, atol=1e-12)
        noise = np.random.default_rng(99123).normal(size=len(classical_series))
        windows = datasets_b.make_windows(noise, 6, 2)
        test_start = round(len(windows) * 0.75) * 2
        unit_power = np.mean(noise[test_start:] ** 2)
        for model in ("QAE", "TinyMLP", "MLP-AE", "LSTM-AE"):
            for sigma in sigmas:
                for row in canonical[(dataset, model, sigma)]:
                    actual = row.get("mse_test_noise", row.get("mse_test_noisy"))
                    assert abs(actual - unit_power * sigma ** 2) < 1e-12
    print("PASS: all four input series and fixed evaluation-noise powers agree.")

    from analyse_qae_floor import fit_floor

    for dataset in datasets:
        points = [(canonical[(dataset, "QAE", sigma)][0]["mse_test_noise"],
                   st.mean(r["mse_test_recovered"]
                           for r in canonical[(dataset, "QAE", sigma)])) for sigma in sigmas]
        intercept, slope = fit_floor(points)
        fitted_slope, fitted_intercept = np.polyfit(*np.array(points).T, 1)
        np.testing.assert_allclose([intercept, slope], [fitted_intercept, fitted_slope],
                                   rtol=1e-12, atol=1e-12)
    print("PASS: affine coefficients independently match numpy least squares.")

    try:
        from scipy.stats import ttest_ind
    except ImportError:
        print("SKIP: scipy unavailable; Welch tests were not checked.")
    else:
        for sigma in sigmas[1:]:
            q = [r["delta_test_pct"] for r in canonical[("finance", "QAE", sigma)]]
            c = [r["delta_test_pct"] for r in canonical[("finance", "TinyMLP", sigma)]]
            result = ttest_ind(c, q, equal_var=False)
            if sigma == 0.35:
                assert round(float(result.statistic), 2) == 2.46
                assert round(float(result.pvalue), 3) == 0.030
            else:
                assert abs(result.statistic) <= 1.78 and result.pvalue >= 0.10
        print("PASS: nine Finance Welch tests match the manuscript's reported pattern.")


if __name__ == "__main__":
    main()
