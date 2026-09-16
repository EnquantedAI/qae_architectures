# project_b_classical — classical AE baselines for the CMES extension

Classical (PyTorch) denoising-autoencoder baselines for the CMES extension of the
QUANTICS 2026 paper ("Co-evolutionary Asymmetry vs. Modular Design: Optimization
Trade-offs in Quantum Denoising Autoencoders"). Matches QUANTICS 2026's own experimental
design exactly (window=6, stride=2, chronological 75/25 split, dynamic per-epoch noise on
the whole series before windowing, Δtest metric) so results are directly comparable to
their Table 2, and to the real eval CSVs from their own repository (`qae_architectures/`,
seed=99123 fixed test noise, same convention).

Two purposes, both requested in the real peer-review of the camera-ready:

- **TinyMLPAutoencoder**: a classical AE parameter-matched to the QAE configs (~140
  params, matching the Monolith QAE's own saturation point) — calibrates the reported
  ~67% noise removal.
- **MLPAutoencoder / LSTMAutoencoder**: a "generous budget" classical comparison
  (thousands of params) — the more usual way people would size a classical denoiser.

Datasets: `mackey_glass_tau17`, `mackey_glass_tau30` (QUANTICS 2026's own chaotic
benchmark, both τ values — τ=17 is the one their reviewers asked to have reintroduced),
`beer` (paper_279/ICCS2024 continuity), `energy`, `finance` (non-chaotic, real-world —
QUANTICS 2026 explicitly did not test any of these).

## Setup

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

No GPU needed — every run here took a few seconds to ~15s on a 2-core cloud CPU.
`torch.cuda.is_available()` is auto-detected (`default_device()` in `classical_ae_b.py`),
so it'll use a GPU automatically if one is present, but it isn't required.

## Running

```bash
# smoke test: 1 seed, sigma=0.2 only, all 3 models, one dataset (few sec/model)
python3 run_classical_grid_b.py --quick --dataset mackey_glass_tau30

# smoke test on all 5 datasets
for d in mackey_glass_tau17 mackey_glass_tau30 beer energy finance; do
  python3 run_classical_grid_b.py --quick --dataset "$d"
done

# full grid: 5 seeds x 3 sigmas (0.1/0.2/0.3) x 3 models x 5 datasets = 225 runs
python3 run_classical_grid_b.py
```

Results accumulate in `results_b/classical_grid_b.json` (resumable — reruns skip rows
already present, so it's safe to Ctrl-C and restart, or split across multiple machines by
running with `--dataset` on each and merging the JSON files afterward). Each row logs to
stdout as it completes:

```
mackey_glass_tau30   sigma=0.2 TinyMLP  seed=2023 n_params=  142 test_R2=+0.763 Delta_test=+40.75% (4.1s)
```

## Viewing / saving the results

`classical_grid_b.json` is raw and not meant to be read directly. Run the summarizer any
time (before, during, or after the grid finishes — it just reports whatever rows exist so
far, and never modifies the JSON):

```bash
python3 summarize_b.py
```

This prints a mean±SD table to the screen (grouped by dataset × model × sigma — the exact
numbers to quote in the paper), and writes three files into `results_b/`:

- `classical_grid_b.csv` — every individual run, one row per (dataset, model, sigma, seed)
  — open this in Excel/Numbers/Google Sheets to look at everything.
- `classical_grid_summary.csv` — the mean±SD table as CSV.
- `classical_grid_summary.txt` — the same table as plain text (identical to what's printed
  to the screen), handy for pasting into notes or an email.

## What to check against the cloud-sandbox run

Re-ran the `--quick --dataset X` smoke test here (1 seed=2023, sigma=0.2) AFTER the
2026-08-30 fixes below. Your run should reproduce these numbers closely (same seed, same
code, deterministic PyTorch ops on CPU — small differences are possible across PyTorch
versions/CPU architectures, but R²/Δtest should match to ~2-3 decimal places). **These
supersede the pre-fix table that used to be here — do not compare old numbers to new ones,
they are not the same computation (see Changelog below).**

| dataset | TinyMLP (~142p) | MLP-AE (~19.2k p) | LSTM-AE (~2.6k p) |
|---|---|---|---|
| mackey_glass_tau30 | R²=0.735, Δtest=+71.40% | R²=0.774, Δtest=+75.54% | R²=0.766, Δtest=+74.68% |
| mackey_glass_tau17 | R²=0.829, Δtest=+82.47% | R²=0.796, Δtest=+79.09% | R²=0.785, Δtest=+77.99% |
| beer | R²=0.288, Δtest=+55.85% | R²=0.264, Δtest=+54.36% | R²=0.261, Δtest=+54.12% |
| energy | R²=0.135, Δtest=+46.23% | R²=0.055, Δtest=+41.25% | R²=-0.036, Δtest=+35.58% |

If your numbers land close to these, the fixes are verified as correct and consistent
across machines. If your full-grid run (all 5 seeds) is what you want to keep for the
paper, that's the one to use — the numbers above are single-seed smoke tests only, meant
for sanity-checking, not for reporting mean±SD in the paper. Note mackey_glass_tau30/
TinyMLP is now Δtest=+71.40%, much closer to the real Monolith QAE's own published 67.15%
than the pre-fix +40.75% was — a good sign the fixes moved things in the right direction
(comparable metric, comparable data), not just "changed the number".

## Changelog — 2026-08-30 (methodology audit fixes)

Before trusting the completed 225-row grid for the paper, this code was audited
against the real repo's own conventions and several real bugs were found. Fixed
here, in order of how much they change the numbers:

1. **Δtest metric was MAE, not MSE** (`metrics_b.py`). The real repo's own `imp_test_pct`
   (published eval CSVs, training notebooks) is unambiguously
   `100*(1-MSE(pred,true)/MSE(noisy,true))` — verified against a real CSV row to 15 decimal
   places. This file used `mean_absolute_error` instead. Biggest single effect on the
   reported percentages.
2. **Mackey-Glass generator had an off-by-one indexing bug** (`datasets_b.py`). The old
   growing-Python-list implementation used `x[-tau]` (negative indexing), which is
   mathematically wrong — should reference the value exactly `tau` steps back via absolute
   indexing on a pre-allocated array, like the real notebook does. Produced a genuinely
   different chaotic series from index=tau onward (max abs diff 0.58–0.76, not a rounding
   error). Also fixed `beta` (0.2→0.17, the real notebook's value, not the paper text's) and
   the downsample offset (`x[::6]`→`normalised[3::6]`, offset 0→3), and reordered
   normalise-then-downsample to match the real notebook exactly.
3. **Model seed didn't control weight initialisation** (`run_classical_grid_b.py`). The
   model object was constructed before `torch.manual_seed(seed)` ran (that call was inside
   `train_dynamic_noise`, called after model construction) — so different `seed` values did
   not actually give independent/reproducible weight initialisations. Fixed by seeding
   before constructing the model in `run_one()`.
4. **Misleading "avoid data leakage" claim** — corrected in `datasets_b.py`'s docstring;
   see the code comment there for the two real (inherited, not "fixed") leakage paths:
   window-boundary sample overlap (stride < window) and whole-series min-max normalisation
   including the test portion. Both should be disclosed as limitations in the paper, since
   they're also present in the real QUANTICS 2026 methodology itself, not unique to this
   reproduction.

All 225 rows from before this changelog entry are invalidated by fix #1 and #2 (different
metric, different underlying data) and need re-running — cheap, ~30-40 min, see Running
above.

## Files

- `datasets_b.py` — Mackey-Glass generation (QUANTICS 2026's exact Forward-Euler recipe,
  τ=17/τ=30), energy/finance/beer loaders, windowing + chronological split.
- `classical_ae_b.py` — `TinyMLPAutoencoder`, `MLPAutoencoder`, `LSTMAutoencoder`;
  `add_noise_to_series` (matches the real repo's `ts_add_noise` exactly: normal noise
  scaled by `sigma * (range_high - range_low)`, `clip=False`); `train_dynamic_noise`
  (whole-series noise resampled every epoch, matching the real training loop found in
  QUANTICS 2026's own source); `eval_fixed_noise` (fixed seed=99123, matching the real
  repo's own eval convention).
- `metrics_b.py` — overlap-averaged window merging (`merge_overlapping_windows`, matches
  the real repo's `ts_wind_flatten_avg`) and the Δtest metric.
- `run_classical_grid_b.py` — the grid driver (see Running, above).
- `data/energy_vic_elec.csv`, `data/finance_aapl.csv` — raw source data for those two
  datasets (Mackey-Glass and beer are generated/embedded in code, no file needed).

## Status (2026-08-29, superseded note)
See also `project_b_qml/` (delivered separately) for real PennyLane Monolith QAE runs —
this classical package only covers the classical baselines.

## Status (2026-08-30)
Full grid (225 rows: 5 datasets x 3 sigmas x 3 models x 5 seeds) re-run in the cloud
sandbox AFTER the 2026-08-30 fixes above — `results_b/classical_grid_b.csv` /
`classical_grid_summary.csv` in this package are these fixed, final numbers. Headline
sanity check: mackey_glass_tau30/TinyMLP/sigma=0.2 -> Delta_test=+70.95% +/- 1.34 (5 seeds),
very close to the real Monolith QAE's own published 67.15% (Table 2) — a good sign the
fixed data/metric pipeline is faithful. The 225-row set the user separately ran on their
own server BEFORE these fixes (via scp'd `project_b_classical.zip`) used the old buggy MAE
metric and the old incorrect Mackey-Glass generator — those numbers should be discarded,
not merged with this run.

## Status (2026-08-29, superseded)
Package delivered to the user and written to `/Users/seba/Desktop/autoencodery/project_b_classical.zip`
on their Mac. Only single-seed `--quick` smoke tests had been run at that point.

## Background: why this exists (noise-semantics fix, 2026-08-29)

An earlier version of this code (built from the QUANTICS 2026 PDF's description alone,
before the real source repo was available) applied noise independently to each window
rather than to the whole series before windowing. That produced a negative Δtest anomaly
(models scoring *worse* than doing nothing) once overlapping windows were averaged for
scoring. Reading the real training code
(`qae_architectures/Jacob/4q_2l_2t_tau17.ipynb::train_stage3_series_noise_simple`) showed
the actual QUANTICS 2026 protocol noises the whole series once per epoch, then windows
it — this version implements that correctly, and the anomaly is gone (all Δtest values
are now positive and consistent with the real QAE results in the same ballpark, see the
project memory / conversation for the full comparison against the real repo's eval CSVs).
