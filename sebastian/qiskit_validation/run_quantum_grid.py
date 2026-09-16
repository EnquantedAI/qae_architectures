"""Run the QuTSAE grid (reduced-scope MVP config per user decision: aw=0,
i.e. 8 qubits = window size exactly, TwoLocal Rx+Ry ansatz, reps=2) across
all three datasets. This is the compute-heavy part - runs as a background
job; ~4.3s/COBYLA-iteration measured on this sandbox's 2-core CPU, so
1000 epochs is ~70 min per dataset, ~3.5h total for all three.

No live MAE tracking during training (that would require per-iteration
reconstruction, prohibitively slow) - only final-parameter reconstruction
and scoring, once training finishes.
"""
import json
import sys
import time

sys.path.insert(0, ".")
import numpy as np

from datasets_new import Target_energy, Target_finance
from qutsae import (QuTSAETrainer, mae_tswin, mape_tswin, r2_tswin,
                     rms_tswin, ts_relang_encode)
from utils.Target import Target_beer
from utils.TS import gen_ts_windows

SAMPLES, SPLIT, WIND, STEP, NOISE = 160, 0.75, 8, 4, 0.03
EPOCHS = 1000
LAT, TRASH, AW, REPS = 7, 1, 0, 2  # MVP config: width=8 (matches window size)

DATASETS = {
    "beer": lambda: Target_beer(),
    "energy": lambda: Target_energy(pt_from=0, pt_to=160),
    "finance": lambda: Target_finance(pt_from=0, pt_to=160),
}

results = {}

for ds_name, ds_fn in DATASETS.items():
    print(f"\n=== {ds_name} ===", flush=True)
    fun = ds_fn()
    _, ytr, _, yva = gen_ts_windows(fun, SAMPLES, SPLIT, WIND, STEP, differencing=True, noise=0)
    _, ytr_n, _, yva_n = gen_ts_windows(fun, SAMPLES, SPLIT, WIND, STEP, differencing=True, noise=NOISE)

    y_train_enc = ts_relang_encode(np.array(ytr))
    y_valid_enc = ts_relang_encode(np.array(yva))
    y_train_noisy_enc = ts_relang_encode(np.array(ytr_n))
    y_valid_noisy_enc = ts_relang_encode(np.array(yva_n))

    trainer = QuTSAETrainer(num_latent=LAT, num_trash=TRASH, aw=AW, reps=REPS,
                             ent="sca", rotation="rxry", shots=2000, seed=2023)
    print(f"width={trainer.width} n_weight_params={len(trainer.weight_params)}", flush=True)

    t0 = time.time()
    fit_res = trainer.fit(
        y_train_noisy_enc, y_train_enc, epochs=EPOCHS, shuffle=True,
        test_ctx=None, seed=2023, log_every=25,
    )
    elapsed = time.time() - t0
    print(f"{ds_name} training done in {elapsed/60:.1f} min, min_cost={fit_res['minimum_cost']:.5f}", flush=True)

    w_opt = fit_res["optimum_parameters"]
    rec_train = trainer.reconstruct(w_opt, y_train_noisy_enc)
    rec_valid = trainer.reconstruct(w_opt, y_valid_noisy_enc)

    def wd(arr):
        return {i: list(arr[i]) for i in range(len(arr))}

    row = {
        "dataset": ds_name, "lat": LAT, "trash": TRASH, "aw": AW, "reps": REPS,
        "n_weight_params": len(trainer.weight_params), "epochs": EPOCHS,
        "elapsed_sec": round(elapsed, 1), "min_cost": fit_res["minimum_cost"],
        "train_r2": r2_tswin(wd(y_train_enc), wd(rec_train)),
        "train_rmse": rms_tswin(wd(y_train_enc), wd(rec_train)),
        "train_mae": mae_tswin(wd(y_train_enc), wd(rec_train)),
        "valid_r2": r2_tswin(wd(y_valid_enc), wd(rec_valid)),
        "valid_rmse": rms_tswin(wd(y_valid_enc), wd(rec_valid)),
        "valid_mae": mae_tswin(wd(y_valid_enc), wd(rec_valid)),
        "objective_func_vals": fit_res["objective_func_vals"],
    }
    results[ds_name] = row
    print(f"{ds_name}: valid_R2={row['valid_r2']:+.3f} valid_MAE={row['valid_mae']:.4f}", flush=True)

    with open("results/quantum_grid.json", "w") as f:
        json.dump(results, f, indent=2)

print("\nAll quantum runs complete. Saved to results/quantum_grid.json", flush=True)
