"""
Track B classical grid - matches QUANTICS 2026's own experimental setup
(window=6, stride=2, dynamic per-iteration noise) so results are directly
comparable to their Table 2 (Monolith: Delta_test=67.15% at this exact
config). Addresses 2 of Reviewer 4's camera-ready suggestions in one run:
  (1) parameter-matched classical baseline (TinyMLPAutoencoder, ~142 params)
  (2) noise-level ablation (their sigma=0.2, plus 0.1 and 0.3)
Plus, beyond their scope: energy/finance (non-chaotic) and beer (paper_279
continuity), and tau=17 Mackey-Glass (reintroduced per Reviewer 4's other ask
-> see project/ Track A for tau=17 already computed there under a different,
paper_279-style pipeline; here it's redone under the QUANTICS 2026 pipeline
for a like-for-like comparison against their tau=30 numbers).

Usage:
    python run_classical_grid_b.py                  # everything (~30-40 min on CPU)
    python run_classical_grid_b.py --quick           # 1 seed, sigma=0.2 only, smoke test
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, ".")
import numpy as np
import torch

from classical_ae_b import (LSTMAutoencoder, MLPAutoencoder, TinyMLPAutoencoder,
                             count_params, eval_fixed_noise, size_for_param_budget,
                             to_window_dict, train_dynamic_noise)
from datasets_b import DATASETS, prepare_dataset
from metrics_b import delta_improvement, window_metrics

WINDOW, STRIDE = 6, 2
LATENT = 4  # matches QUANTICS 2026 Table 2's 4/2 latent-trash split (6 qubits)
SEEDS_DEFAULT = [2023, 2024, 2025, 2026, 2027]
SIGMAS_DEFAULT = [0.1, 0.2, 0.3]  # 0.2 = their value; 0.1/0.3 = reviewer-requested ablation
EPOCHS = 300

TINY_HIDDEN = size_for_param_budget(WINDOW, LATENT, target_params=140)

MODELS = {
    "TinyMLP": lambda: TinyMLPAutoencoder(WINDOW, LATENT, hidden=TINY_HIDDEN),
    "MLP-AE": lambda: MLPAutoencoder(WINDOW, LATENT, hidden=(88, 66, 44)),
    "LSTM-AE": lambda: LSTMAutoencoder(WINDOW, LATENT, hidden_size=16),
}

OUT_PATH = "results_b/classical_grid_b.json"


def run_one(ds_name, sigma, model_name, seed):
    _, _, series = prepare_dataset(ds_name, window_size=WINDOW, stride=STRIDE)
    # FIXED 2026-08-30: model must be constructed AFTER seeding, not before -
    # torch.manual_seed(seed) inside train_dynamic_noise() ran too late to
    # control this model's weight initialisation (nn.Linear/nn.LSTM init
    # their weights at construction time), so `seed` was not actually
    # controlling replication independence/reproducibility as intended.
    torch.manual_seed(seed)
    model = MODELS[model_name]()
    n_params = count_params(model)

    t0 = time.time()
    train_dynamic_noise(model, series, sigma=sigma, window=WINDOW, stride=STRIDE,
                         train_frac=0.75, epochs=EPOCHS, seed=seed, log_every=0)
    elapsed = time.time() - t0

    # fixed test-noise seed matches the real repo's convention (99123), so this
    # is directly comparable to the real eval CSVs' own numbers
    noisy_test, pred_test, test_clean = eval_fixed_noise(
        model, series, sigma=sigma, window=WINDOW, stride=STRIDE, train_frac=0.75)
    m = window_metrics(test_clean, pred_test, stride=STRIDE)
    delta, mse_noisy, mse_pred = delta_improvement(noisy_test, pred_test, test_clean, stride=STRIDE)

    return {
        "dataset": ds_name, "model": model_name, "sigma": sigma, "seed": seed,
        "window": WINDOW, "stride": STRIDE, "latent": LATENT,
        "n_params": n_params, "epochs": EPOCHS, "elapsed_sec": round(elapsed, 2),
        "test_r2": m["r2"], "test_rmse": m["rmse"], "test_mae": m["mae"],
        "delta_test_pct": delta,
        # saved so future re-derivation of delta_test_pct doesn't require retraining:
        "mse_test_noisy": mse_noisy, "mse_test_recovered": mse_pred,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="1 seed, sigma=0.2 only (smoke test)")
    ap.add_argument("--dataset", choices=list(DATASETS), default=None)
    args = ap.parse_args()

    seeds = [2023] if args.quick else SEEDS_DEFAULT
    sigmas = [0.2] if args.quick else SIGMAS_DEFAULT
    datasets = [args.dataset] if args.dataset else list(DATASETS)

    os.makedirs("results_b", exist_ok=True)
    print(f"TinyMLP hidden size for ~140-param budget (window={WINDOW}, latent={LATENT}): "
          f"{TINY_HIDDEN} -> {count_params(MODELS['TinyMLP']())} params")
    print(f"Grid: datasets={datasets} sigmas={sigmas} seeds={seeds} models={list(MODELS)}")

    results = []
    if os.path.exists(OUT_PATH):
        results = json.load(open(OUT_PATH))
    done_keys = {(r["dataset"], r["sigma"], r["model"], r["seed"]) for r in results}

    for ds_name in datasets:
        for sigma in sigmas:
            for model_name in MODELS:
                for seed in seeds:
                    key = (ds_name, sigma, model_name, seed)
                    if key in done_keys:
                        continue
                    row = run_one(ds_name, sigma, model_name, seed)
                    results.append(row)
                    with open(OUT_PATH, "w") as f:
                        json.dump(results, f, indent=2)
                    print(f"{ds_name:20s} sigma={sigma:.1f} {model_name:8s} seed={seed} "
                          f"n_params={row['n_params']:5d} test_R2={row['test_r2']:+.3f} "
                          f"Delta_test={row['delta_test_pct']:+.2f}% ({row['elapsed_sec']:.1f}s)",
                          flush=True)

    print(f"\nDone. {len(results)} rows in {OUT_PATH}")


if __name__ == "__main__":
    main()
