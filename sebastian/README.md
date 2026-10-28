# QAE journal extension: code and recorded results

This standalone package contains the current journal-extension experiments,
including the four-series noise ablation and the reconstruction-error analysis.
The raw experiment files were synchronized on 28 October 2026. No new training
was performed during that synchronization, and the recorded MSEs and positive-noise
Delta values were not modified.

## Extension of the QUANTICS 2026 conference paper

This package supports the journal extension of the QUANTICS 2026 conference
paper, which primarily compared quantum autoencoder architectures. The extension
adds parameter-matched classical autoencoders with training-partition
hyperparameter selection, additional time series, a denser noise-level ablation
including zero-noise reconstruction measurements, and an empirical analysis of
reconstruction error and estimated break-even noise levels. The package contains
the code, raw results, and derived artifacts supporting these additions.

## Experimental tracks

- `pennylane_monolith/`: six-sample windows, four latent qubits, four circuit
  layers, 144 parameters, 300 epochs. Four series: Mackey--Glass at tau 17 and
  tau 30, Victoria electricity demand, and AAPL closing prices. Ten noise
  levels: 0 and 0.10 through 0.50 in steps of 0.05; seven seeds, 2023--2029.
  The raw file contains 280 canonical runs and two 20-epoch smoke runs;
  aggregation excludes the smoke runs.
- `classical_baselines/`: TinyMLP (142 parameters), MLP-AE, and LSTM-AE;
  training-partition hyperparameter search at sigma=0.2 and final evaluation
  at the same ten noise levels. Search records, selected configurations, raw
  final runs, and the older untuned baseline are included. Raw final logs
  also contain earlier configurations and exploratory seeds; canonical
  aggregation selects the recorded winning configuration and seeds 2023--2029.
- `qiskit_validation/`: independent QuTSAE validation, four series and five
  seeds. Its encoding, preprocessing, training objective, and metrics differ;
  do not pool these results with the matched PennyLane benchmark. Its experiment
  code and per-seed results were retained from the earlier package.

## Reproduce the recorded analysis

Use an existing Python 3.11+ environment with numpy, pandas and matplotlib.
The unchanged per-track `requirements.txt` files describe the training stacks;
no training framework is required to aggregate the stored JSON files.
From `sebastian/`:

```bash
python analysis/check_seed_counts.py
python analysis/verify_tracks_match.py
python analysis/verify_results.py
python analysis/aggregate_comparison.py
python analysis/analyse_qae_floor.py > results_tables/qae_floor_analysis.txt
python pennylane_monolith/summarize_qml.py
python analysis/generate_extension_figures.py
```

`verify_tracks_match.py` checks the real-world loaders and, when PennyLane is
available, the Mackey--Glass training generator. `verify_results.py` additionally
checks the numerical generator bodies without importing the training frameworks.
It verifies raw metrics, canonical seed uniqueness, aggregation, input-series
agreement, and the current comparison CSV. Welch tests run when scipy is available.
Figures are rebuilt as PNG and PDF under `figures/`; all paths are package-local.
The canonical comparison table covers 36 series/noise cells per model, with
seven runs per cell. Derived summaries use population SD (`ddof=0`), matching
this manuscript; Welch tests use sample variances.

## Reproduce training

After setting up the appropriate per-track environments, run the following from
`sebastian/`. Both scripts resume missing configurations; do not run concurrent
writers against the same raw JSON file.

```bash
PY=/path/to/pennylane-env/bin/python WORKERS=4 bash pennylane_monolith/run_all_qml.sh
PY=/path/to/torch-env/bin/python WORKERS=4 bash classical_baselines/run_all_classical.sh
```

These commands may be expensive. Analysis-only commands above do not train models.
The runners preserve the recorded protocol; synchronization does not rerun training.

## Interpretation and known limitations

- Delta = 100 * (1 - MSE_reconstructed / MSE_noisy), evaluated after averaging
  overlapping reconstructions. Delta is undefined at sigma=0. Some historical
  QAE rows contain enormous negative placeholder percentages there, and other
  rows contain a zero sentinel. The raw logs are preserved for provenance;
  comparison tables and Delta plots exclude zero noise, and summaries mark
  it undefined. The valid zero-noise MSE remains in the affine fit.
- The fitted intercept phi is distinct from measured zero-noise reconstruction
  error. Models are trained separately at each noise level. The affine fit and
  its break-even estimates describe this protocol, not a proven irreducible
  architectural bound. PCA residual variance is exploratory, not a four-qubit
  capacity bound or an independently validated predictor.
- Scaling uses the full series and neighboring train/test windows overlap.
  This is a controlled within-series experiment, not a purged forecasting test.
- Seven seeds characterize training variability on one fixed test-noise draw;
  they do not provide seven independent test series or test-noise realizations.
- The JSON logs do not contain trained weights or per-sample predictions.
  Aggregation is reproducible from these logs; independently checking the
  model-to-MSE step requires retraining or separately supplied checkpoints.

`results_tables/raw_result_manifest.json` records SHA-256 hashes and row counts
of the synchronized raw JSON files. Source logs are preserved including their
noncanonical runs, which makes the selection applied by the aggregator auditable.
