"""Run the full classical-baseline grid (MLP-AE + LSTM-AE, lat=1..8) across
all three datasets (beer=ICCS baseline, energy, finance=new MVP). Fast
(seconds per config), so runs in the foreground."""
import json
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

DATASETS = {
    "beer": lambda: Target_beer(),
    "energy": lambda: Target_energy(pt_from=0, pt_to=160),
    "finance": lambda: Target_finance(pt_from=0, pt_to=160),
    "mackey_glass": lambda: Target_mackey_glass(pt_from=0, pt_to=160, tau=17),
}

results = []

for ds_name, ds_fn in DATASETS.items():
    fun = ds_fn()
    _, ytr, _, yva = gen_ts_windows(fun, SAMPLES, SPLIT, WIND, STEP, differencing=True, noise=0)
    _, ytr_n, _, yva_n = gen_ts_windows(fun, SAMPLES, SPLIT, WIND, STEP, differencing=True, noise=NOISE)
    ytr, yva, ytr_n, yva_n = map(np.array, (ytr, yva, ytr_n, yva_n))

    for lat in range(1, 9):
        for model_name, build in [
            ("MLP-AE", lambda lat=lat: MLPAutoencoder(WIND, lat, hidden=(88, 66, 44))),
            ("LSTM-AE", lambda lat=lat: LSTMAutoencoder(WIND, lat, hidden_size=16)),
        ]:
            model = build()
            n_params = count_params(model)
            t0 = time.time()
            hist = train_classical_ae(
                model, ytr_n, ytr, yva_n, yva, epochs=300, lr=2e-3, batch_size=8,
                log_every=0, verbose=False, seed=2023,
            )
            elapsed = time.time() - t0

            rec_train = reconstruct(model, ytr_n)
            rec_valid = reconstruct(model, yva_n)

            row = {
                "dataset": ds_name, "model": model_name, "lat": lat, "n_params": n_params,
                "elapsed_sec": round(elapsed, 2),
                "train_r2": r2_tswin(to_window_dict(ytr), to_window_dict(rec_train)),
                "train_rmse": rms_tswin(to_window_dict(ytr), to_window_dict(rec_train)),
                "train_mae": mae_tswin(to_window_dict(ytr), to_window_dict(rec_train)),
                "valid_r2": r2_tswin(to_window_dict(yva), to_window_dict(rec_valid)),
                "valid_rmse": rms_tswin(to_window_dict(yva), to_window_dict(rec_valid)),
                "valid_mae": mae_tswin(to_window_dict(yva), to_window_dict(rec_valid)),
                "final_train_mse": hist["train_losses"][-1],
                "final_valid_mse": hist["valid_losses"][-1],
            }
            results.append(row)
            print(f"{ds_name:8s} {model_name:8s} lat={lat} params={n_params:5d} "
                  f"valid_R2={row['valid_r2']:+.3f} valid_MAE={row['valid_mae']:.4f} "
                  f"({elapsed:.1f}s)")

with open("results/classical_grid.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\nSaved {len(results)} rows to results/classical_grid.json")
