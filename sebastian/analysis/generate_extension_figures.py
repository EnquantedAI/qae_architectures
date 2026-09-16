"""Generate the two publication figures for the isolated extension package.

The primary figure compares only the parameter-matched PennyLane and classical
experiments.  The Qiskit/Aer figure is intentionally separate because its
preprocessing and metrics differ from the primary benchmark.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = ROOT / "figures"
CLASSICAL_SUMMARY = ROOT / "classical_baselines" / "results_b" / "classical_grid_summary.csv"
QML_SUMMARY = ROOT / "pennylane_monolith" / "results_qml" / "monolith_qml_summary.csv"
QISKIT_RESULTS = ROOT / "qiskit_validation" / "results" / "quantum_full"
QISKIT_CLASSICAL = ROOT / "qiskit_validation" / "results" / "classical_grid_multiseed.json"

COLORS = {
    "TinyMLP": "#0072B2",
    "Monolith QAE": "#D55E00",
    "QuTSAE": "#D55E00",
    "MLP-AE": "#0072B2",
    "LSTM-AE": "#009E73",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read a CSV file into a list of row dictionaries."""
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def mean_and_std(values: list[float]) -> tuple[float, float]:
    """Return population mean and standard deviation, matching stored summaries."""
    array = np.asarray(values, dtype=float)
    return float(array.mean()), float(array.std())


def make_primary_benchmark_figure() -> None:
    """Plot the parameter-matched TinyMLP versus PennyLane Monolith sweep."""
    classical_rows = read_csv(CLASSICAL_SUMMARY)
    qml_rows = read_csv(QML_SUMMARY)

    classical = {
        (int(row["dataset"].removeprefix("mackey_glass_tau")), float(row["sigma"])): row
        for row in classical_rows
        if row["model"] == "TinyMLP" and row["dataset"].startswith("mackey_glass_tau")
    }
    qml = {(int(row["tau"]), float(row["sigma"])): row for row in qml_rows}

    sigmas = [0.1, 0.2, 0.3]
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.8), sharey=True, layout="constrained")

    for axis, tau in zip(axes, (17, 30), strict=True):
        classical_mean = [float(classical[tau, sigma]["delta_test_pct_mean"]) for sigma in sigmas]
        classical_std = [float(classical[tau, sigma]["delta_test_pct_std"]) for sigma in sigmas]
        qml_mean = [float(qml[tau, sigma]["delta_test_pct_mean"]) for sigma in sigmas]
        qml_std = [float(qml[tau, sigma]["delta_test_pct_std"]) for sigma in sigmas]

        axis.errorbar(
            sigmas, classical_mean, yerr=classical_std, marker="o", capsize=3,
            color=COLORS["TinyMLP"], label="TinyMLP (142 parameters)",
        )
        axis.errorbar(
            sigmas, qml_mean, yerr=qml_std, marker="s", capsize=3,
            color=COLORS["Monolith QAE"], label="Monolith QAE (144 parameters)",
        )
        axis.set_title(rf"Mackey--Glass $\tau={tau}$")
        axis.set_xlabel(r"Noise level $\sigma$")
        axis.set_xticks(sigmas)
        axis.set_xlim(0.08, 0.32)
        axis.grid(axis="y", alpha=0.25)

    axes[0].set_ylabel(r"Denoising improvement $\Delta_{test}$ (\%)")
    axes[0].set_ylim(0, 100)
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles, labels, loc="lower right", frameon=True)

    FIGURES_DIR.mkdir(exist_ok=True)
    for extension in ("png", "pdf"):
        fig.savefig(FIGURES_DIR / f"matched_pennylane_classical_benchmark.{extension}", dpi=300)
    plt.close(fig)


def make_qiskit_validation_figure() -> None:
    """Plot the independent Qiskit/Aer validation without pooling protocols."""
    quantum_rows = []
    for path in sorted(QISKIT_RESULTS.glob("*_seed*.json")):
        with path.open() as handle:
            quantum_rows.append(json.load(handle))
    with QISKIT_CLASSICAL.open() as handle:
        classical_rows = json.load(handle)

    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in quantum_rows:
        grouped[(row["dataset"], "QuTSAE")].append(float(row["valid_r2"]))
    for row in classical_rows:
        grouped[(row["dataset"], row["model"])].append(float(row["valid_r2"]))

    datasets = ["beer", "energy", "finance", "mackey_glass"]
    labels = ["Beer", "Energy", "Finance", r"Mackey--Glass ($\tau=17$)"]
    models = ["QuTSAE", "MLP-AE", "LSTM-AE"]
    x_positions = np.arange(len(datasets))
    width = 0.24

    fig, axis = plt.subplots(figsize=(8.2, 4.1), layout="constrained")
    for index, model in enumerate(models):
        means, stds = zip(*(mean_and_std(grouped[(dataset, model)]) for dataset in datasets))
        axis.bar(
            x_positions + (index - 1) * width, means, width, yerr=stds, capsize=3,
            color=COLORS[model], label=model,
        )

    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_ylabel(r"Validation $R^2$")
    axis.set_xticks(x_positions, labels)
    axis.set_ylim(-0.55, 1.1)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False, ncol=3, loc="upper center")
    axis.set_title("Independent Qiskit/Aer validation (five seeds per model)")

    FIGURES_DIR.mkdir(exist_ok=True)
    for extension in ("png", "pdf"):
        fig.savefig(FIGURES_DIR / f"qiskit_validation_r2.{extension}", dpi=300)
    plt.close(fig)


def main() -> None:
    """Generate all extension figures from versioned result artifacts."""
    make_primary_benchmark_figure()
    make_qiskit_validation_figure()


if __name__ == "__main__":
    main()
