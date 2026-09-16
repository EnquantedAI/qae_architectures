"""
Noise-level ablation sweep for the Monolith QAE, using the real training code
in train_monolith_qae.py - this is Reviewer 4's camera-ready ask #4 ("1-2
additional noise levels beyond sigma=0.2"), which has NO existing data in the
real repo's eval CSVs (those only cover sigma=0.2) - so these are genuinely
new numbers, not a re-run of something already computed.

Scope: Mackey-Glass ONLY (tau=17 and tau=30) - same as QUANTICS 2026's own
paper, which never tested the QAE on beer/energy/finance. Those three only
exist in the classical comparison (project_b_classical); there's no reviewer
request or existing precedent for running the quantum model on them.

Each run is essentially single-core-bound (lightning.qubit on an 8-wire
circuit is dominated by per-circuit-call Python/C++ dispatch overhead, not
raw FLOPs) - so it does NOT get faster just because the machine has many
cores/RAM; only running several INDEPENDENT runs at once, in separate
processes, uses those extra cores. That's what --workers does: it runs
multiple (tau, sigma, seed) combinations concurrently via
ProcessPoolExecutor, each pinned to 1 CPU thread internally (see
_worker_init()) so they don't also fight each other for BLAS/OMP threads.

Each run takes ~15-20 min on ONE FREE CPU core (300 epochs,
window=6/n_latent=4/n_layers=4/Rxyz -> 8 wires) - that estimate assumes
nothing else CPU-heavy is running on the same machine. If something else is
(e.g. a Qiskit/COBYLA grid), every run here slows down roughly in proportion
to how much CPU that other process is using - that's contention, not a bug,
and more workers won't fix it (they'll make it worse) while the machine is
still busy with something else.

The full default grid (2 tau x 3 sigma x 3 seeds = 18 runs) is ~4.5-6 hours
of TOTAL compute; with e.g. --workers 6 on an otherwise-idle many-core
machine that becomes roughly ~1 hour of wall-clock time.

Usage:
    # smoke test: 1 seed, 20 epochs, both tau, all 3 sigmas (a few min total)
    python3 sweep_monolith_qae.py --quick

    # the real grid, sequential (safe default, 1 worker)
    python3 sweep_monolith_qae.py

    # the real grid, 6 runs in parallel - use this once the machine is
    # actually free (e.g. after a Qiskit/COBYLA grid on the same server has
    # finished); keep --workers comfortably below the machine's free core
    # count
    python3 sweep_monolith_qae.py --workers 6

    # just the two new noise levels (skip sigma=0.2, already covered by the
    # real repo's own eval CSVs at every architecture) - two calls, 12 runs
    # total instead of 18
    python3 sweep_monolith_qae.py --sigma 0.1
    python3 sweep_monolith_qae.py --sigma 0.3

Resumable: reruns skip (tau, sigma, seed, n_epochs) combinations already
present in results_qml/monolith_qml.json, so Ctrl-C + restart, or splitting
work across multiple --sigma/--tau calls (even on different machines, then
merging the JSON files), is safe.
"""
import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

TAUS_DEFAULT = [17, 30]
SIGMAS_DEFAULT = [0.1, 0.2, 0.3]  # 0.2 = their published value; 0.1/0.3 = the ablation
SEEDS_DEFAULT = [2023, 2024, 2025, 2026, 2027]
N_EPOCHS_DEFAULT = 300

OUT_PATH = "results_qml/monolith_qml.json"


def _worker_init():
    """ProcessPoolExecutor initializer - runs once per worker process, BEFORE
    that process imports numpy/pennylane (those imports happen lazily inside
    _run_one_worker below, not at module level) - so these env vars are
    actually in effect when BLAS/OMP first initialise, on both 'fork' and
    'spawn' platforms. Setting them after numpy is already imported (e.g. if
    this were done post-fork with numpy imported at module level) does
    nothing - BLAS reads thread-count env vars once, at its own init time."""
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = "1"


def _run_one_worker(tau, sigma, seed, n_epochs):
    from train_monolith_qae import run_one  # lazy: see _worker_init's docstring
    t0 = time.time()
    row = run_one(tau, sigma, seed, n_epochs)
    row["elapsed_sec"] = round(time.time() - t0, 1)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="1 seed, 20 epochs, smoke test")
    ap.add_argument("--tau", type=int, choices=[17, 30], default=None)
    ap.add_argument("--sigma", type=float, default=None)
    ap.add_argument("--workers", type=int, default=1,
                     help="how many runs to compute concurrently (separate processes). "
                          "Keep this comfortably below the machine's free core count.")
    args = ap.parse_args()

    taus = [args.tau] if args.tau else TAUS_DEFAULT
    sigmas = [args.sigma] if args.sigma else SIGMAS_DEFAULT
    seeds = [2025] if args.quick else SEEDS_DEFAULT
    n_epochs = 20 if args.quick else N_EPOCHS_DEFAULT

    os.makedirs("results_qml", exist_ok=True)
    results = json.load(open(OUT_PATH)) if os.path.exists(OUT_PATH) else []
    done_keys = {(r["tau"], r["sigma"], r["seed"], r["n_epochs"]) for r in results}

    todo = [(tau, sigma, seed) for tau in taus for sigma in sigmas for seed in seeds
            if (tau, sigma, seed, n_epochs) not in done_keys]

    print(f"Grid: tau={taus} sigma={sigmas} seeds={seeds} n_epochs={n_epochs} workers={args.workers}")
    print(f"({len(taus) * len(sigmas) * len(seeds)} combination(s) requested, "
          f"{len(done_keys)} already done overall, {len(todo)} to do now)\n")

    if args.workers <= 1:
        for tau, sigma, seed in todo:
            row = _run_one_worker(tau, sigma, seed, n_epochs)
            results.append(row)
            with open(OUT_PATH, "w") as f:
                json.dump(results, f, indent=2)
            print(f"tau={tau} sigma={sigma:.1f} seed={seed} n_weights={row['n_weights']} "
                  f"Delta_test={row['delta_test_pct']:+.2f}% ({row['elapsed_sec']:.0f}s)", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers, initializer=_worker_init) as pool:
            futures = {pool.submit(_run_one_worker, tau, sigma, seed, n_epochs): (tau, sigma, seed)
                       for tau, sigma, seed in todo}
            for fut in as_completed(futures):
                row = fut.result()
                results.append(row)
                with open(OUT_PATH, "w") as f:
                    json.dump(results, f, indent=2)
                print(f"tau={row['tau']} sigma={row['sigma']:.1f} seed={row['seed']} "
                      f"n_weights={row['n_weights']} Delta_test={row['delta_test_pct']:+.2f}% "
                      f"({row['elapsed_sec']:.0f}s)", flush=True)

    print(f"\nDone. {len(results)} row(s) in {OUT_PATH}")


if __name__ == "__main__":
    main()
