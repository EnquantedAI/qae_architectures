# Sebastian's extension package

This directory is an isolated, reproducible extension of the QAE architecture
study.  It deliberately does not modify the shared implementation elsewhere in
the repository.

## Scope

The package contains three distinct experimental tracks:

1. `pennylane_monolith/` reproduces the fresh PennyLane Monolith QAE sweep on
   Mackey--Glass series with $\tau \in \{17, 30\}$ and
   $\sigma \in \{0.1, 0.2, 0.3\}$.
2. `classical_baselines/` contains parameter-matched classical autoencoders
   evaluated with the same final MSE-based $\Delta_{test}$ protocol.  The
   TinyMLP has 142 parameters and the Monolith QAE has 144 parameters.
3. `qiskit_validation/` contains the independent Qiskit/Aer QuTSAE validation
   and its MLP-AE/LSTM-AE baselines on four time series.

The first two tracks form the primary, directly matched benchmark.  The Qiskit
track has different preprocessing, windowing and metrics, and must therefore
be reported separately rather than numerically pooled with the primary result.

## Contents

- `classical_baselines/`: source code, raw input data, final raw results and
  summaries for the MSE-based baseline sweep.
- `pennylane_monolith/`: source code, utility functions, final raw results and
  summaries for the PennyLane sweep.
- `qiskit_validation/`: Qiskit/Aer source code, raw input data and complete
  per-seed final results.  The obsolete `_summary.*` files are intentionally
  absent; `aggregate_results.py` rebuilds them directly from the 20 per-seed
  JSON files.
- `analysis/generate_extension_figures.py`: deterministic generator for the
  two figures used by the extension.
- `figures/`: generated PNG and PDF figures.

The runner scripts retain their native output locations: `results_b/` for the
classical track, `results_qml/` for the PennyLane track, and `results/` for the
Qiskit/Aer track.

## Reproducing the reported summaries

Each experiment has its own `requirements.txt`, because PennyLane and the
Qiskit/Aer GPU stack have separate environment constraints.  From the relevant
subdirectory:

```bash
python -m pip install -r requirements.txt
python summarize_b.py                 # classical_baselines
python summarize_qml.py               # pennylane_monolith
python aggregate_results.py            # qiskit_validation
```

The expensive runners are `run_classical_grid_b.py`, `sweep_monolith_qae.py`,
and `run_full_grid_gpu.py`, respectively.  They are resumable and preserve
per-run JSON outputs.

To regenerate the publication figures, install `numpy` and `matplotlib` in an
environment that can read CSV/JSON files, then run from this directory:

```bash
python analysis/generate_extension_figures.py
```

## Reporting constraint

The primary matched sweep shows that the TinyMLP achieves a higher mean
$\Delta_{test}$ in all six matched $(\tau, \sigma)$ settings.  The independent
Qiskit/Aer validation is heterogeneous across datasets.  These artifacts
support a qualified architectural conclusion inside the investigated QAE
family; they do not support a general claim of quantum superiority over
classical autoencoders.
