# project_b_qml — real Monolith QAE, PennyLane, for the noise-level ablation

This is the first genuinely NEW thing to run on PennyLane for this project — everything
delivered before this was classical (`project_b_classical`) or a re-run of the old
Qiskit/COBYLA pipeline (Track A, `qutsae_gpu_package.zip`).

**What it is**: the real Monolith QAE — the QUANTICS 2026 paper's best/simplest
architecture — extracted directly from the real source notebook
`qae_architectures/Jacob/ts_pl_v6_03_monolith_mcgen_tau30_jc_q6_l4_t2_x2_r4_0_pi4pi4_clean.ipynb`
into a headless, parameterized script. The ansatz (`full_qae`), the cost function, and the
training loop (`train_with_noise`) are lifted essentially verbatim from that notebook — same
math, just no plotting/debug cells, plus a CLI and JSON logging.

**Why this exists / what's actually new**: the real repo's eval CSVs
(`qae_architectures/Jacob/qae_eval_framework/eval_all_mackey_glass_tau{17,30}_n200.csv`)
already have real Monolith/Sidekick/Two-stage results — but only at **sigma=0.2**, the
one noise level QUANTICS 2026 published. Reviewer 4's camera-ready request #4 ("add 1-2
additional noise levels beyond sigma=0.2") has **no existing data anywhere in the repo** —
this script computes those, using their real training code, so the ablation numbers are
authoritative rather than reimplemented-from-scratch guesses.

## Setup

The exact version pins in `requirements.txt` matter — see the comment at the top of that
file for why (pip's current default autograd/autoray resolution breaks PennyLane 0.40.0
in two different ways; both are fixed by these two pins). If you already have a working
PennyLane environment from the real `qae_architectures` repo itself, that should also work
and you can skip this.

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

No GPU needed — this runs on `lightning.qubit` (CPU simulator), same as the real repo's
own default. ~3s/epoch on a single CPU core for the headline config (window=6,
n_latent=4, n_layers=4, Rxyz -> 8 wires including trash-qubit ancillas, 144 trainable
weights — matches the paper's own ~140-parameter saturation point almost exactly).

## Running

```bash
# smoke test: one run, 20 epochs (~1 min) - checks everything works
python3 train_monolith_qae.py --tau 30 --sigma 0.2 --seed 2025 --n_epochs 20

# one full-length run at their exact published config (~15-20 min on CPU)
python3 train_monolith_qae.py --tau 30 --sigma 0.2 --seed 2025

# the noise-ablation grid Reviewer 4 asked for: tau x {17,30} x sigma x {0.1,0.2,0.3}
# x 3 seeds, 300 epochs each = 18 runs, ~4.5-6 hours total - fine to leave running
# overnight/unattended, it's resumable (skips rows already in results_qml/monolith_qml.json)
python3 sweep_monolith_qae.py

# quick sanity sweep first (1 seed, 20 epochs, all tau x sigma - a few min)
python3 sweep_monolith_qae.py --quick

# view/save results any time (before, during, or after the sweep finishes)
python3 summarize_qml.py
```

`summarize_qml.py` prints a mean±SD table (grouped by tau x sigma) and writes
`results_qml/monolith_qml.csv` (every run) and `results_qml/monolith_qml_summary.csv`/`.txt`
(the summary), same pattern as `project_b_classical/summarize_b.py`.

## Validation done so far (2026-08-29, in the cloud sandbox)

Ran the real training loop for 60/300 epochs (20% of full length) on tau=30, sigma=0.2,
seed=2025: cost dropped from 0.105 -> 0.011 and was still decreasing, Delta_test was
already +44.27% at 20% of training — a plausible trajectory toward the paper's own
published ~67% at full 300 epochs (which this partial run wasn't long enough to reach).
Parameter count matched exactly (144 weights, vs. the paper's own reported ~140-parameter
saturation point for this config). This is a strong signal the extraction is faithful to
the real notebook, but **no full-length (300-epoch) run has been completed and compared
against the paper's Table 2 headline number yet** — worth doing once as a sanity check
before trusting the ablation grid's numbers for the paper.

## Files

- `train_monolith_qae.py` — the real ansatz/training/eval code (see docstring at the top
  for exactly which notebook cells this was extracted from) + CLI for one run.
- `sweep_monolith_qae.py` — the tau x sigma x seed grid driver.
- `summarize_qml.py` — results viewer/exporter.
- `qae_utils/Window.py` — copied unchanged from the real repo (`ts_add_noise`,
  `ts_wind_make`, `ts_wind_split`, `ts_wind_flatten_avg` — the noise/windowing primitives
  this script's `create_sw_tens` is built on).

## Known open items

- Only the Monolith architecture is implemented here (the best/simplest performer, and
  the one with a clean single-file real training notebook to extract from). Mirror,
  Stacked, and Sidekick's real training code is more involved (multi-stage, teacher
  forcing / SWAP-test pretraining) — not attempted yet. Given the real eval CSVs already
  have Sidekick/Two-stage(Stacked) numbers at sigma=0.2, and Monolith is the paper's
  headline result, this was the highest-value one to get running first.
- Training noise uses `noise_clip=True` (matches this specific notebook's own training
  setting). Eval noise now uses `noise_clip=False` — see Changelog below — these are
  deliberately different, not inconsistent: training-time convention vs. the official
  eval-framework convention.

## Changelog — 2026-08-30 (methodology audit fix: eval protocol)

The user's own audit found a real comparability problem: this script's **eval** noise used
`seed=seed+n_epochs` (different every run) and `noise_clip=True`, while
`project_b_classical`'s eval — and, critically, the REAL "official" evaluation framework
that actually produced the published eval CSVs / Table 2 numbers
(`Jacob/qae_eval_framework/universal_framework_all.ipynb`) — both use a **fixed**
`seed=99123` and `clip=False`. Any run computed before this fix has a Δtest that is not
directly comparable across seeds/epoch counts, nor to `project_b_classical`'s numbers.

Fixed in `run_one()`: eval noise (the `create_sw_tens(...)` call used for the final
Δtest/mse_test_* computation, AFTER training finishes) now uses `seed=99123, noise_clip=False`
unconditionally, matching the official framework and `project_b_classical` exactly.
**Training-time noise is unchanged** (`noise_clip=True`, still resampled each iteration per
the real training loop) — only the final evaluation convention was standardised, since
that's the part that needs to be comparable across runs and across tracks. The result dict
now separately reports `train_noise_clip` and `eval_seed_fixed`/`eval_noise_clip` so this
is auditable from the JSON output itself, not just from reading the code.

Any rows already computed with the old eval convention (e.g. if you started
`sweep_monolith_qae.py --workers N` before this fix landed) should be treated as
diagnostic-only and re-run — `results_qml/monolith_qml.json` is resumable/keyed on
`(tau, sigma, seed, n_epochs)`, so it can't tell old-convention rows apart from new ones by
itself; safest is to delete `results_qml/monolith_qml.json` and start the sweep fresh with
this fixed code.
