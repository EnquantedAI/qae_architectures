"""
Noise-level ablation sweep for the Monolith QAE, using the real training code
in train_monolith_qae.py - this is Reviewer 4's camera-ready ask #4 ("1-2
additional noise levels beyond sigma=0.2"), which has NO existing data in the
real repo's eval CSVs (those only cover sigma=0.2) - so these are genuinely
new numbers, not a re-run of something already computed.

Scope (extended 2026-09-08): Mackey-Glass (tau=17, tau=30) AND the real-world,
non-chaotic series `energy` and `finance`, via --dataset.

The real-world half is new. QUANTICS 2026 tested the QAE on Mackey-Glass only,
so the classical-vs-quantum comparison rested on a single synthetic dataset;
the classical side already covered energy/finance under this exact protocol
(experiments/classical/results_b/classical_grid_b.json), the quantum side did
not. `beer` is deliberately NOT included: it is paper_279's own dataset, and
the QuTSAE configuration it would be compared against was designed on it -
a home-field advantage that makes that particular comparison uninformative.

Everything except the data-loading block is shared across datasets (encoding
range, windowing, training loop, noise schedule, fixed eval seed 99123,
Delta_test metric), and datasets_qml.py mirrors the classical track's loaders
line-for-line - run verify_tracks_match.py to assert both tracks see
numerically identical series before trusting any cross-track table.

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

    # NEW: the real-world series, full noise sweep, 5 seeds (15 runs each)
    python3 sweep_monolith_qae.py --dataset energy  --workers 6
    python3 sweep_monolith_qae.py --dataset finance --workers 6

    # smoke test on a real-world series first (1 seed, 20 epochs, ~1 min)
    python3 sweep_monolith_qae.py --dataset energy --quick

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


def _run_one_worker(tau, sigma, seed, n_epochs, dataset="mackey_glass"):
    from train_monolith_qae import run_one  # lazy: see _worker_init's docstring
    t0 = time.time()
    row = run_one(tau, sigma, seed, n_epochs, dataset=dataset)
    row["elapsed_sec"] = round(time.time() - t0, 1)
    return row


def _row_key(row):
    """Resume key. Rows written before the dataset flag existed have no
    'dataset' field and are, by construction, Mackey-Glass - treat them as
    such so the 30 existing MG rows are still recognised as done and never
    recomputed."""
    return (row.get("dataset", "mackey_glass"), row.get("tau"), row["sigma"],
            row["seed"], row["n_epochs"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="1 seed, 20 epochs, smoke test")
    ap.add_argument("--dataset", default="mackey_glass",
                     help="mackey_glass (default, sweeps --tau) or a real-world "
                          "series: energy, finance. For real-world series --tau "
                          "is ignored (recorded as null in the results).")
    ap.add_argument("--tau", type=int, choices=[17, 30], default=None)
    ap.add_argument("--sigma", type=float, default=None)
    ap.add_argument("--sigmas", default=None,
                     help="comma-separated noise levels, e.g. '0,0.1,0.15' (overrides --sigma); "
                          "one call runs all missing jobs in a single worker pool")
    ap.add_argument("--seeds", default=None,
                     help="comma-separated seeds, e.g. '2023,2024'. Default: "
                          f"{','.join(str(s) for s in SEEDS_DEFAULT)}")
    ap.add_argument("--workers", type=int, default=1,
                     help="how many runs to compute concurrently (separate processes). "
                          "Keep this comfortably below the machine's free core count.")
    args = ap.parse_args()

    # Mackey-Glass is the only dataset with a tau; for the real-world series a
    # single [None] "tau" keeps the loop below dataset-agnostic.
    if args.dataset == "mackey_glass":
        taus = [args.tau] if args.tau is not None else TAUS_DEFAULT
    else:
        if args.tau is not None:
            print(f"note: --tau is ignored for dataset={args.dataset}")
        taus = [None]
    # `is not None`, not truthiness: --sigma 0 is a legitimate request (it
    # measures the reconstruction floor on the clean series) and `if args.sigma`
    # would silently discard it and launch the whole default 3-sigma sweep.
    if args.sigmas:
        sigmas = [float(x) for x in args.sigmas.split(",") if x.strip()]
    else:
        sigmas = [args.sigma] if args.sigma is not None else SIGMAS_DEFAULT
    if args.seeds:
        seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    else:
        seeds = [2025] if args.quick else SEEDS_DEFAULT
    n_epochs = 20 if args.quick else N_EPOCHS_DEFAULT

    os.makedirs("results_qml", exist_ok=True)
    results = json.load(open(OUT_PATH)) if os.path.exists(OUT_PATH) else []
    done_keys = {_row_key(r) for r in results}

    todo = [(tau, sigma, seed) for tau in taus for sigma in sigmas for seed in seeds
            if (args.dataset, tau, sigma, seed, n_epochs) not in done_keys]

    print(f"Grid: dataset={args.dataset} tau={taus} sigma={sigmas} seeds={seeds} "
          f"n_epochs={n_epochs} workers={args.workers}")
    print(f"({len(taus) * len(sigmas) * len(seeds)} combination(s) requested, "
          f"{len(done_keys)} already done overall, {len(todo)} to do now)\n")

    def _log(row):
        tau_s = "  -" if row.get("tau") is None else f"{row['tau']:3d}"
        print(f"{row.get('dataset', 'mackey_glass'):13s} tau={tau_s} sigma={row['sigma']:.2f} "
              f"seed={row['seed']} n_weights={row['n_weights']} "
              f"Delta_test={row['delta_test_pct']:+.2f}% ({row['elapsed_sec']:.0f}s)", flush=True)

    if args.workers <= 1:
        for tau, sigma, seed in todo:
            row = _run_one_worker(tau, sigma, seed, n_epochs, args.dataset)
            results.append(row)
            with open(OUT_PATH, "w") as f:
                json.dump(results, f, indent=2)
            _log(row)
    else:
        with ProcessPoolExecutor(max_workers=args.workers, initializer=_worker_init) as pool:
            futures = {pool.submit(_run_one_worker, tau, sigma, seed, n_epochs, args.dataset):
                       (tau, sigma, seed) for tau, sigma, seed in todo}
            for fut in as_completed(futures):
                row = fut.result()
                results.append(row)
                with open(OUT_PATH, "w") as f:
                    json.dump(results, f, indent=2)
                _log(row)

    print(f"\nDone. {len(results)} row(s) in {OUT_PATH}")


if __name__ == "__main__":
    main()
