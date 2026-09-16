"""Multi-seed classical baseline runs at lat=7 (the QuTSAE paper's best
latent size) across the 3 datasets, for a proper mean +/- SD comparison
against the quantum grid (run_full_grid_gpu.py uses the same seed list).
Cheap on CPU - runs fine in the dev sandbox, doesn't need the GPU machine."""
import json
import os
import sys
import time

sys.path.insert(0, ".")
import numpy as np

from classical_ae import (LSTMAutoencoder, MLPAutoencoder, count_params,
                           reconstruct, to_window_dict, train_classical_ae)
from datasets_new import Target_energy, Target_finance, Target_mackey_glass
from qutsae import mae_tswin, mape_tswin, r2_tswin, rms_tswin
from utils.Target import Target_beer
from utils.TS import gen_ts_windows

SAMPLES, SPLIT, WIND, STEP, NOISE = 160, 0.75, 8, 4, 0.03
LAT = 7
SEEDS = [2023, 2024, 2025, 2026, 2027]

DATASETS = {
    "beer": lambda: Target_beer(),
    "energy": lambda: Target_energy(pt_from=0, pt_to=160),
    "finance": lambda: Target_finance(pt_from=0, pt_to=160),
    "mackey_glass": lambda: Target_mackey_glass(pt_from=0, pt_to=160, tau=17),
}

os.makedirs("results", exist_ok=True)
results = []

for ds_name, ds_fn in DATASETS.items():
    fun = ds_fn()
    _, ytr, _, yva = gen_ts_windows(fun, SAMPLES, SPLIT, WIND, STEP, differencing=True, noise=0)
    _, ytr_n, _, yva_n = gen_ts_windows(fun, SAMPLES, SPLIT, WIND, STEP, differencing=True, noise=NOISE)
    ytr, yva, ytr_n, yva_n = map(np.array, (ytr, yva, ytr_n, yva_n))

    for seed in SEEDS:
        for model_name, build in [
            ("MLP-AE", lambda: MLPAutoencoder(WIND, LAT, hidden=(88, 66, 44))),
            ("LSTM-AE", lambda: LSTMAutoencoder(WIND, LAT, hidden_size=16)),
        ]:
            model = build()
            t0 = time.time()
            hist = train_classical_ae(
                model, ytr_n, ytr, yva_n, yva, epochs=300, lr=2e-3, batch_size=8,
                log_every=0, verbose=False, seed=seed,
            )
            elapsed = time.time() - t0
            rec_train = reconstruct(model, ytr_n)
            rec_valid = reconstruct(model, yva_n)
            row = {
                "dataset": ds_name, "model": model_name, "lat": LAT, "seed": seed,
                "n_params": count_params(model), "elapsed_sec": round(elapsed, 2),
                "train_r2": r2_tswin(to_window_dict(ytr), to_window_dict(rec_train)),
                "train_rmse": rms_tswin(to_window_dict(ytr), to_window_dict(rec_train)),
                "train_mae": mae_tswin(to_window_dict(ytr), to_window_dict(rec_train)),
                "valid_r2": r2_tswin(to_window_dict(yva), to_window_dict(rec_valid)),
                "valid_rmse": rms_tswin(to_window_dict(yva), to_window_dict(rec_valid)),
                "valid_mae": mae_tswin(to_window_dict(yva), to_window_dict(rec_valid)),
            }
            results.append(row)
            print(f"{ds_name:8s} {model_name:8s} seed={seed} "
                  f"valid_R2={row['valid_r2']:+.3f} valid_MAE={row['valid_mae']:.4f}")

with open("results/classical_grid_multiseed.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved {len(results)} rows to results/classical_grid_multiseed.json")
