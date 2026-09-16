"""Run this FIRST on the GPU machine, before the multi-hour full grid, to
confirm: (1) the environment is set up correctly, (2) GPU is actually being
picked up by qiskit-aer, (3) a full training loop runs end-to-end and the
cost decreases. Takes ~1-3 minutes."""
import sys
import time

sys.path.insert(0, ".")
import numpy as np
import torch

from qutsae import QuTSAETrainer, detect_device, ts_relang_encode
from classical_ae import MLPAutoencoder, default_device, train_classical_ae
from utils.Target import Target_beer
from utils.TS import gen_ts_windows

print("=" * 60)
print("ENVIRONMENT CHECK")
print("=" * 60)
print("PyTorch CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("  GPU:", torch.cuda.get_device_name(0))
print("Qiskit-Aer device detected:", detect_device())
print("classical_ae default_device():", default_device())

print("\n" + "=" * 60)
print("SMALL QuTSAE TRAINING RUN (lat=7, trash=1, aw=3, reps=2, "
      "the paper's exact best config; 30 iterations only)")
print("=" * 60)

_, ytr, _, yva = gen_ts_windows(Target_beer(), 160, 0.75, 8, 4, differencing=True, noise=0)
_, ytr_n, _, yva_n = gen_ts_windows(Target_beer(), 160, 0.75, 8, 4, differencing=True, noise=0.03)
y_train_enc = ts_relang_encode(np.array(ytr))
y_train_noisy_enc = ts_relang_encode(np.array(ytr_n))

trainer = QuTSAETrainer(num_latent=7, num_trash=1, aw=3, reps=2, rotation="rxry", shots=2000)
t0 = time.time()
res = trainer.fit(y_train_noisy_enc, y_train_enc, epochs=30, test_ctx=None, log_every=10)
elapsed = time.time() - t0
actual_iters = len(res["objective_func_vals"])
per_iter = elapsed / actual_iters
if actual_iters != 30:
    print(f"\n[note] requested epochs=30, but COBYLA actually ran {actual_iters} "
          f"iterations - it enforces a MINIMUM of (n_weight_params + 2) function "
          f"evaluations regardless of the requested count (n_weight_params="
          f"{res['n_weight_params']} here, for the aw=3/11-qubit ansatz), so a "
          f"small smoke-test request gets silently bumped up. This is a smoke-test-"
          f"only quirk: the real grid's epochs=2000 is already well above that floor, "
          f"so it is unaffected and will run exactly 2000 iterations as requested.")
print(f"\n{actual_iters} iterations took {elapsed:.1f}s => {per_iter:.2f} s/iter")
print(f"Estimated time for full 2000-epoch run: {per_iter*2000/60:.0f} min")
print(f"Estimated time for the full grid (4 datasets x 5 seeds = 20 runs): "
      f"{per_iter*2000*20/3600:.1f} h")
print(f"Cost trend: first={res['objective_func_vals'][0]:.4f} "
      f"last={res['objective_func_vals'][-1]:.4f} "
      f"(should generally decrease over 30 iters, though COBYLA can be noisy this early)")

print("\nIf the per-iter time above is a few seconds or less, the full grid "
      "in run_full_grid_gpu.py should comfortably finish within hours, not "
      "days. If it's still >10s/iter, GPU is probably not actually being "
      "used - check the 'device' lines above.")
