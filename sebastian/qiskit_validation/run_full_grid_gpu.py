"""
FULL-FIDELITY QuTSAE grid — for the GPU environment.

Paper-faithful config: lat=7, trash=1, aw=3, reps=2 (11 qubits, TwoLocal
Rx+Ry ansatz, matching the ICCS 2024 paper's best model), 2000 COBYLA
epochs, x3 datasets (beer=ICCS baseline, energy, finance=new MVP datasets),
x N_SEEDS random seeds for a mean +/- SD comparison against the classical
baselines (statistical rigor requested in the CMES extension plan).

On the 2-core CPU-only dev sandbox this config measured ~4.3s/COBYLA-
iteration => ~17h for ONE run; that's why this script was deferred to a
GPU machine. Runs automatically on GPU if `qiskit-aer-gpu` + a working CUDA
install are present (see detect_device() in qutsae.py) - falls back to CPU
otherwise (just very slow).

Usage:
    python run_full_grid_gpu.py                  # all datasets, all seeds
    python run_full_grid_gpu.py --dataset energy  # single dataset
    python run_full_grid_gpu.py --seeds 2023 2024 # subset of seeds
    python run_full_grid_gpu.py --epochs 200      # quick smoke test override

Two-GPU (A40 + L4) parallel use: see run_two_gpu.sh, which launches two
instances of this script concurrently, each pinned to one card via
CUDA_VISIBLE_DEVICES and given a disjoint slice of the 5 seeds. Each run
writes its own results/quantum_full/{dataset}_seed{seed}.json - safe under
concurrent writers since the filenames never collide. There is NO shared
summary file written here (that used to be a read-modify-write race when
two processes wrote it at once); run aggregate_results.py afterwards (or
any time, even mid-run) to rebuild results/quantum_full/_summary.json and
results/quantum_full/_summary.csv from whatever per-run files exist so far.
"""
import argparse
import json
import os
import time

import numpy as np

from datasets_new import Target_energy, Target_finance, Target_mackey_glass
from qutsae import (QuTSAETrainer, mae_tswin, mape_tswin, r2_tswin,
                     rms_tswin, ts_relang_encode, detect_device)
from utils.Target import Target_beer
from utils.TS import gen_ts_windows

SAMPLES, SPLIT, WIND, STEP, NOISE = 160, 0.75, 8, 4, 0.03
EPOCHS_DEFAULT = 2000
LAT, TRASH, AW, REPS = 7, 1, 3, 2  # <-- paper's exact best config (Table 1/2)
SEEDS_DEFAULT = [2023, 2024, 2025, 2026, 2027]

# beer/energy/finance = non-chaotic (real-world); mackey_glass (tau=17) =
# synthetic chaotic reference - included to show QuTSAE generalises across
# both regimes, not just non-chaotic. See datasets_new.py for details.
DATASETS = {
    "beer": lambda: Target_beer(),
    "energy": lambda: Target_energy(pt_from=0, pt_to=160),
    "finance": lambda: Target_finance(pt_from=0, pt_to=160),
    "mackey_glass": lambda: Target_mackey_glass(pt_from=0, pt_to=160, tau=17),
}

OUT_DIR = "results/quantum_full"


def wd(arr):
    return {i: list(arr[i]) for i in range(len(arr))}


def run_one(ds_name, seed, epochs):
    fun = DATASETS[ds_name]()
    _, ytr, _, yva = gen_ts_windows(fun, SAMPLES, SPLIT, WIND, STEP, differencing=True, noise=0)
    _, ytr_n, _, yva_n = gen_ts_windows(fun, SAMPLES, SPLIT, WIND, STEP, differencing=True, noise=NOISE)

    y_train_enc = ts_relang_encode(np.array(ytr))
    y_valid_enc = ts_relang_encode(np.array(yva))
    y_train_noisy_enc = ts_relang_encode(np.array(ytr_n))
    y_valid_noisy_enc = ts_relang_encode(np.array(yva_n))

    trainer = QuTSAETrainer(num_latent=LAT, num_trash=TRASH, aw=AW, reps=REPS,
                             ent="sca", rotation="rxry", shots=2000, seed=seed)

    t0 = time.time()
    fit_res = trainer.fit(
        y_train_noisy_enc, y_train_enc, epochs=epochs, shuffle=True,
        test_ctx=None, seed=seed, log_every=50,
    )
    elapsed = time.time() - t0

    w_opt = fit_res["optimum_parameters"]
    rec_train = trainer.reconstruct(w_opt, y_train_noisy_enc)
    rec_valid = trainer.reconstruct(w_opt, y_valid_noisy_enc)

    return {
        "dataset": ds_name, "seed": seed, "lat": LAT, "trash": TRASH, "aw": AW, "reps": REPS,
        "n_weight_params": len(trainer.weight_params), "epochs": epochs,
        "device": trainer.device, "elapsed_sec": round(elapsed, 1),
        "min_cost": fit_res["minimum_cost"],
        "train_r2": r2_tswin(wd(y_train_enc), wd(rec_train)),
        "train_rmse": rms_tswin(wd(y_train_enc), wd(rec_train)),
        "train_mae": mae_tswin(wd(y_train_enc), wd(rec_train)),
        "valid_r2": r2_tswin(wd(y_valid_enc), wd(rec_valid)),
        "valid_rmse": rms_tswin(wd(y_valid_enc), wd(rec_valid)),
        "valid_mae": mae_tswin(wd(y_valid_enc), wd(rec_valid)),
        "objective_func_vals": fit_res["objective_func_vals"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(DATASETS), default=None,
                     help="Run a single dataset only (default: all)")
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS_DEFAULT)
    ap.add_argument("--epochs", type=int, default=EPOCHS_DEFAULT)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    datasets = [args.dataset] if args.dataset else list(DATASETS)

    print(f"Device detected: {detect_device()}")
    print(f"Grid: datasets={datasets} seeds={args.seeds} epochs={args.epochs} "
          f"config=lat{LAT}_trash{TRASH}_aw{AW}_reps{REPS}")

    # NOTE: done-checking is purely filesystem-based (per-(dataset,seed) output
    # file), NOT based on a shared summary file. This is what makes it safe to
    # run two instances of this script concurrently (see run_two_gpu.sh) -
    # each process only ever touches files whose names it alone writes, so
    # there is no read-modify-write race. Run aggregate_results.py separately
    # (any time, even while runs are still in progress) to build/refresh the
    # combined summary.

    for ds_name in datasets:
        for seed in args.seeds:
            out_path = os.path.join(OUT_DIR, f"{ds_name}_seed{seed}.json")
            if os.path.exists(out_path):
                print(f"[skip, already done] {ds_name} seed={seed}")
                continue
            print(f"\n=== {ds_name} seed={seed} ===", flush=True)
            row = run_one(ds_name, seed, args.epochs)
            with open(out_path, "w") as f:
                json.dump(row, f, indent=2)
            print(f"{ds_name} seed={seed}: valid_R2={row['valid_r2']:+.3f} "
                  f"valid_MAE={row['valid_mae']:.4f} ({row['elapsed_sec']/60:.1f} min)", flush=True)

    print("\nAll requested runs complete. Run `python aggregate_results.py` to "
          "rebuild the combined summary.")


if __name__ == "__main__":
    main()
