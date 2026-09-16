"""
Properly-tuned classical baselines for the CMES extension.

THE PROBLEM THIS SOLVES
-----------------------
`run_classical_grid_b.py` trains every classical model with ONE fixed
hyperparameter setting (300 epochs, lr=2e-3, LSTM hidden=16), identical
across all datasets. That is defensible as "no per-dataset tuning", but it is
not symmetric with how the quantum side was produced: the QAE configuration
reported in QUANTICS 2026 was itself selected from a sweep over qubit counts,
latent/trash splits and 1-5 circuit layers. A reviewer can therefore say -
correctly - that the classical baseline never got the same courtesy, and that
any classical underperformance may just be an untuned baseline.

Evidence that this is a real effect, not a hypothetical: under the fixed
setting, the LSTM-AE reaches only train R^2 = 0.25-0.51 on some series, i.e.
it fails to fit the TRAINING data at all. That is underfitting, not a
property of the data.

WHAT THIS SCRIPT DOES
---------------------
Stage 1 (search). For each (dataset, model), at the reference noise level
sigma=0.2 only, train every configuration in that model's grid over
SEARCH_SEEDS and score it on the TRAINING partition. The best mean training
score wins.

Stage 2 (final). Re-run the winning configuration over all noise levels and
all FINAL_SEEDS, and report the standard test-set Delta_test (fixed eval
noise seed 99123, clip=False) - the same protocol as every other number in
the comparison.

THREE PROPERTIES THAT MAKE THIS DEFENSIBLE
------------------------------------------
1. Selection never touches the test partition. Stage-1 rows contain training
   metrics ONLY - the test metrics are not even computed there, so there is
   no path by which test data could influence which configuration is chosen.
   The selection noise realisation (SELECT_SEED) also differs from every
   noise realisation seen during training, so what is being scored is
   denoising of unseen noise on seen signal, not memorisation.
2. Hyperparameters are selected ONCE per (dataset, model) at sigma=0.2, then
   applied unchanged at every noise level. Nothing is re-tuned per condition,
   so the noise-level ablation stays an ablation rather than 3 independent
   tunings.
3. The parameter-matched baseline stays parameter-matched. TinyMLP's grid
   varies ONLY optimisation hyperparameters (epochs, learning rate); its
   capacity is pinned at the ~142-parameter budget that matches the QAE. The
   "generous budget" models (MLP-AE, LSTM-AE) are allowed to vary capacity,
   because for them the point was never parameter matching.

The baseline (untuned) configuration is included in every grid, so the output
also answers "how much did tuning actually change?" - if the answer is "not
much", that is itself a result worth reporting, and it retires the
"your classical baseline was just undertrained" objection either way.

Usage:
    python3 run_classical_tuned_b.py --quick                       # smoke test (~2 min)
    python3 run_classical_tuned_b.py --datasets energy,finance --workers 4
    python3 run_classical_tuned_b.py                               # everything

Resumable: completed rows are keyed and skipped on re-run, so Ctrl-C and
restart is safe.
"""
import argparse
import itertools
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

WINDOW, STRIDE = 6, 2
LATENT = 4                      # QUANTICS 2026 Table 2: 6 qubits, 4/2 latent-trash
TRAIN_FRAC = 0.75
BATCH_SIZE = 8

SIGMA_SELECT = 0.2              # the paper's own reference noise level
SIGMAS_FINAL = [0.1, 0.2, 0.3]
SEARCH_SEEDS = [2023, 2024, 2025]
FINAL_SEEDS = [2023, 2024, 2025, 2026, 2027, 2028, 2029]

# Noise realisation used to SCORE configurations on the training partition.
# Must differ from the eval seed (99123) and from every training-epoch seed
# (seed*100_000 + epoch) so that selection sees noise the model never trained on.
SELECT_SEED = 424242

DATASETS_DEFAULT = ["energy", "finance"]

SEARCH_OUT = "results_b/classical_tuned_search.json"
FINAL_OUT = "results_b/classical_tuned_final.json"
SELECTED_OUT = "results_b/classical_tuned_selected.json"


# ---------------------------------------------------------------------------
# Configuration grids
# ---------------------------------------------------------------------------
# TinyMLP: capacity is FIXED (parameter-matched to the QAE). Only optimisation
# hyperparameters vary - this is what removes the "undertrained baseline"
# objection without weakening the parameter-matching claim.
GRID_TINY = [{"epochs": e, "lr": lr}
             for e, lr in itertools.product([300, 1000], [1e-3, 2e-3, 5e-3])]

# MLP-AE / LSTM-AE: the "generous budget" references - capacity may vary.
GRID_MLP = [{"hidden": h, "epochs": e, "lr": lr}
            for h, e, lr in itertools.product([(88, 66, 44), (32, 24, 16)],
                                              [300, 1000], [1e-3, 2e-3])]
GRID_LSTM = [{"hidden_size": hs, "epochs": e, "lr": lr}
             for hs, e, lr in itertools.product([16, 32, 64],
                                                [300, 1000], [1e-3, 2e-3])]

CONFIG_GRIDS = {"TinyMLP": GRID_TINY, "MLP-AE": GRID_MLP, "LSTM-AE": GRID_LSTM}

# What run_classical_grid_b.py used - present in every grid above, so the
# tuned result is always directly comparable to the untuned one.
BASELINE_CONFIG = {
    "TinyMLP": {"epochs": 300, "lr": 2e-3},
    "MLP-AE": {"hidden": (88, 66, 44), "epochs": 300, "lr": 2e-3},
    "LSTM-AE": {"hidden_size": 16, "epochs": 300, "lr": 2e-3},
}


def _cfg_key(cfg):
    """Stable, JSON-friendly identity for a config dict."""
    return json.dumps({k: list(v) if isinstance(v, tuple) else v
                       for k, v in sorted(cfg.items())}, sort_keys=True)


def build_model(model_name, cfg):
    from classical_ae_b import (LSTMAutoencoder, MLPAutoencoder, TinyMLPAutoencoder,
                                 size_for_param_budget)
    if model_name == "TinyMLP":
        hidden = size_for_param_budget(WINDOW, LATENT, target_params=140)
        return TinyMLPAutoencoder(WINDOW, LATENT, hidden=hidden)
    if model_name == "MLP-AE":
        return MLPAutoencoder(WINDOW, LATENT, hidden=tuple(cfg["hidden"]))
    if model_name == "LSTM-AE":
        return LSTMAutoencoder(WINDOW, LATENT, hidden_size=cfg["hidden_size"])
    raise KeyError(model_name)


# ---------------------------------------------------------------------------
# One run
# ---------------------------------------------------------------------------

def _evaluate(model, series, sigma, partition, noise_seed):
    """Delta / R2 on one partition ('train' or 'test') under a single fixed
    noise realisation. Mirrors classical_ae_b.eval_fixed_noise, but lets the
    caller pick the partition and the noise seed - needed because model
    selection must score the TRAIN partition under a noise realisation that
    is neither a training draw nor the reporting one."""
    import numpy as np
    import torch

    from classical_ae_b import NOISE_RANGE, add_noise_to_series
    from datasets_b import make_windows
    from metrics_b import delta_improvement, window_metrics

    series = np.asarray(series, dtype=float)
    clean_w = make_windows(series, WINDOW, STRIDE)
    n_train = int(round(len(clean_w) * TRAIN_FRAC))
    sl = slice(0, n_train) if partition == "train" else slice(n_train, None)

    noisy_w = make_windows(add_noise_to_series(series, sigma, seed=noise_seed,
                                               lo=NOISE_RANGE[0], hi=NOISE_RANGE[1]),
                           WINDOW, STRIDE)
    clean_part, noisy_part = clean_w[sl], noisy_w[sl]

    model.eval()
    with torch.no_grad():
        device = next(model.parameters()).device
        rec = model(torch.tensor(noisy_part, dtype=torch.float32, device=device)).cpu().numpy()

    m = window_metrics(clean_part, rec, stride=STRIDE)
    delta, mse_noisy, mse_rec = delta_improvement(noisy_part, rec, clean_part, stride=STRIDE)
    return {"delta_pct": delta, "r2": m["r2"], "rmse": m["rmse"],
            "mse_noisy": mse_noisy, "mse_recovered": mse_rec}


def run_one(dataset, model_name, cfg, sigma, seed, stage):
    """Train one model. stage='search' returns TRAIN metrics only (the test
    partition is never evaluated); stage='final' returns the reported test
    metrics (plus train metrics for reference)."""
    import torch

    from classical_ae_b import count_params, train_dynamic_noise
    from datasets_b import prepare_dataset

    _, _, series = prepare_dataset(dataset, window_size=WINDOW, stride=STRIDE)

    # seed BEFORE constructing the model: nn.Linear/nn.LSTM initialise their
    # weights at construction time (the 2026-08-30 seeding-order fix)
    torch.manual_seed(seed)
    model = build_model(model_name, cfg)

    t0 = time.time()
    train_dynamic_noise(model, series, sigma=sigma, window=WINDOW, stride=STRIDE,
                        train_frac=TRAIN_FRAC, epochs=cfg["epochs"], lr=cfg["lr"],
                        batch_size=BATCH_SIZE, seed=seed, log_every=0)
    elapsed = round(time.time() - t0, 2)

    row = {"stage": stage, "dataset": dataset, "model": model_name,
           "config": {k: list(v) if isinstance(v, tuple) else v for k, v in cfg.items()},
           "config_key": _cfg_key(cfg), "sigma": sigma, "seed": seed,
           "window": WINDOW, "stride": STRIDE, "latent": LATENT,
           "n_params": count_params(model), "elapsed_sec": elapsed}

    tr = _evaluate(model, series, sigma, "train", SELECT_SEED)
    row.update({"select_delta_pct": tr["delta_pct"], "select_r2": tr["r2"],
                "select_noise_seed": SELECT_SEED})

    if stage == "final":
        from classical_ae_b import TEST_SEED_FIXED
        te = _evaluate(model, series, sigma, "test", TEST_SEED_FIXED)
        row.update({"delta_test_pct": te["delta_pct"], "test_r2": te["r2"],
                    "test_rmse": te["rmse"], "mse_test_noisy": te["mse_noisy"],
                    "mse_test_recovered": te["mse_recovered"],
                    "eval_noise_seed": TEST_SEED_FIXED})
    return row


def _worker_init():
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = "1"


def _worker(args):
    return run_one(*args)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _load(path):
    return json.load(open(path)) if os.path.exists(path) else []


def _save(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(rows, f, indent=2)


def _key(row):
    return (row["stage"], row["dataset"], row["model"], row["config_key"],
            row["sigma"], row["seed"])


def _execute(jobs, workers, existing, out_path, label):
    """Run `jobs` (tuples for run_one), appending to `existing`, saving after
    each completion so the run is resumable."""
    rows = list(existing)
    done = {_key(r) for r in rows}
    todo = [j for j in jobs
            if (j[5], j[0], j[1], _cfg_key(j[2]), j[3], j[4]) not in done]
    print(f"{label}: {len(jobs)} job(s) requested, {len(jobs) - len(todo)} already done, "
          f"{len(todo)} to run")
    if not todo:
        return rows

    t0 = time.time()
    if workers <= 1:
        for i, job in enumerate(todo, 1):
            row = _worker(job)
            rows.append(row)
            _save(out_path, rows)
            _progress(i, len(todo), row, t0)
    else:
        with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init) as pool:
            futures = [pool.submit(_worker, job) for job in todo]
            for i, fut in enumerate(as_completed(futures), 1):
                row = fut.result()
                rows.append(row)
                _save(out_path, rows)
                _progress(i, len(todo), row, t0)
    return rows


def _progress(i, total, row, t0):
    el = time.time() - t0
    eta = el / i * (total - i)
    scored = (f"Delta_test={row['delta_test_pct']:+6.2f}%" if "delta_test_pct" in row
              else f"select_Delta={row['select_delta_pct']:+6.2f}%")
    print(f"  [{i:3d}/{total}] {row['dataset']:8s} {row['model']:8s} "
          f"sigma={row['sigma']:.1f} seed={row['seed']} "
          f"{row['config']} p={row['n_params']:<6d} {scored} "
          f"({row['elapsed_sec']:.1f}s, ETA {eta/60:.1f} min)", flush=True)


def select_best(search_rows, datasets, models):
    """Mean training-partition Delta over SEARCH_SEEDS decides the winner.
    Returns {(dataset, model): (config, summary_dict)}."""
    import statistics as st

    chosen = {}
    for ds in datasets:
        for mdl in models:
            by_cfg = {}
            for r in search_rows:
                if r["dataset"] == ds and r["model"] == mdl and r["stage"] == "search":
                    by_cfg.setdefault(r["config_key"], []).append(r)
            if not by_cfg:
                continue
            scored = []
            for key, rs in by_cfg.items():
                deltas = [r["select_delta_pct"] for r in rs]
                scored.append({
                    "config_key": key, "config": rs[0]["config"],
                    "n_params": rs[0]["n_params"], "n_seeds": len(deltas),
                    "select_delta_mean": sum(deltas) / len(deltas),
                    "select_delta_sd": st.stdev(deltas) if len(deltas) > 1 else 0.0,
                    "select_r2_mean": sum(r["select_r2"] for r in rs) / len(rs),
                })
            scored.sort(key=lambda s: -s["select_delta_mean"])
            best = scored[0]
            baseline_key = _cfg_key(BASELINE_CONFIG[mdl])
            baseline = next((s for s in scored if s["config_key"] == baseline_key), None)
            chosen[(ds, mdl)] = (best["config"], {
                "dataset": ds, "model": mdl, "selected": best,
                "baseline": baseline, "n_configs": len(scored),
                "ranking": scored,
            })
    return chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default=",".join(DATASETS_DEFAULT),
                    help="comma-separated (energy,finance,mackey_glass_tau17,"
                         "mackey_glass_tau30,beer)")
    ap.add_argument("--models", default="TinyMLP,MLP-AE,LSTM-AE")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--quick", action="store_true",
                    help="1 seed, 2 configs/model, sigma=0.2 only - smoke test")
    ap.add_argument("--search-only", action="store_true")
    args = ap.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    grids = {m: CONFIG_GRIDS[m] for m in models}
    search_seeds, final_seeds, sigmas = SEARCH_SEEDS, FINAL_SEEDS, SIGMAS_FINAL
    if args.quick:
        grids = {m: g[:2] for m, g in grids.items()}
        search_seeds, final_seeds, sigmas = [2023], [2023], [0.2]

    print(f"Datasets: {datasets}\nModels:   {models}")
    print(f"Search:   sigma={SIGMA_SELECT}, seeds={search_seeds}, "
          f"{sum(len(g) for g in grids.values())} configs/dataset, "
          f"scored on the TRAIN partition (noise seed {SELECT_SEED})")
    print(f"Final:    sigmas={sigmas}, seeds={final_seeds}, "
          f"test Delta (noise seed 99123)\n")

    # ---- Stage 1: search -------------------------------------------------
    jobs = [(ds, mdl, cfg, SIGMA_SELECT, seed, "search")
            for ds in datasets for mdl in models
            for cfg in grids[mdl] for seed in search_seeds]
    search_rows = _execute(jobs, args.workers, _load(SEARCH_OUT), SEARCH_OUT, "Stage 1 (search)")

    chosen = select_best(search_rows, datasets, models)
    # MERGE, don't overwrite: a later run restricted to other datasets (e.g.
    # --datasets mackey_glass_tau17,mackey_glass_tau30) must not wipe the
    # selections recorded for energy/finance - aggregate_comparison.py uses
    # this file to decide which config is canonical, and a missing entry
    # silently re-admits smoke-test rows into the means.
    merged = {(i["dataset"], i["model"]): i for i in _load(SELECTED_OUT)}
    merged.update({k: info for k, (_, info) in chosen.items()})
    _save(SELECTED_OUT, list(merged.values()))

    print("\n--- selected configurations (by mean TRAIN Delta) ---")
    for (ds, mdl), (cfg, info) in sorted(chosen.items()):
        b, base = info["selected"], info["baseline"]
        delta_note = ""
        if base is not None:
            gain = b["select_delta_mean"] - base["select_delta_mean"]
            delta_note = (f"  | baseline {base['config']} -> "
                          f"{base['select_delta_mean']:+.2f}%  (tuning gain {gain:+.2f} pp)")
        print(f"{ds:8s} {mdl:8s} {cfg}  p={b['n_params']}  "
              f"train Delta {b['select_delta_mean']:+.2f}%{delta_note}")
    print(f"\nSelection detail written to {SELECTED_OUT}")

    if args.search_only:
        return

    # ---- Stage 2: final --------------------------------------------------
    print()
    final_jobs = [(ds, mdl, chosen[(ds, mdl)][0], sigma, seed, "final")
                  for ds in datasets for mdl in models if (ds, mdl) in chosen
                  for sigma in sigmas for seed in final_seeds]
    _execute(final_jobs, args.workers, _load(FINAL_OUT), FINAL_OUT, "Stage 2 (final)")
    print(f"\nDone. Search rows -> {SEARCH_OUT}, final rows -> {FINAL_OUT}")


if __name__ == "__main__":
    main()
