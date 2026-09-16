"""
Cross-track consistency check: the classical track (classical_baselines) and
the quantum track (pennylane_monolith) MUST see numerically identical series.

Why this file exists
--------------------
The 2026-08-30 audit found that the two tracks had silently drifted apart -
different Mackey-Glass generators, a different Delta_test metric, and a
different eval-noise protocol - which made every cross-track number in the
comparison table meaningless until fixed. Adding the real-world series
(energy, finance) to the quantum track re-opens exactly that risk: two copies
of a loader that must agree to the last bit.

This script is the guard. Run it after touching either loader, and before
building any table that puts classical and quantum numbers side by side.

    python3 verify_tracks_match.py

Exits non-zero (and prints what differs) if anything drifts. It only needs
numpy/pandas - no torch, no PennyLane - so it can run anywhere.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
CLASSICAL = os.path.join(PARENT, "classical_baselines")
QUANTUM = os.path.join(PARENT, "pennylane_monolith")

for path in (CLASSICAL, QUANTUM):
    if not os.path.isdir(path):
        sys.exit(f"FAIL: expected to find {path} next to this script")

# Import the two tracks' loaders under distinct names. Both packages have
# module-level files with different names (datasets_b vs datasets_qml), so a
# plain sys.path insert of both directories is unambiguous.
sys.path.insert(0, CLASSICAL)
sys.path.insert(0, QUANTUM)

import datasets_b            # noqa: E402  (classical track)
import datasets_qml          # noqa: E402  (quantum track)

# The quantum track's Mackey-Glass lives in train_monolith_qae.py, which
# imports PennyLane. Import it only if available, so this script still runs
# (and still checks the real-world series) in a plain numpy environment.
try:
    import train_monolith_qae as qml_train
    HAVE_PENNYLANE = True
except Exception as exc:  # pragma: no cover - environment-dependent
    HAVE_PENNYLANE = False
    _import_error = exc


REALWORLD = ["energy", "finance"]
TAUS = [17, 30]


# Two different bars, deliberately:
#
#   * Real-world series: the two loaders are a line-for-line mirror of each
#     other (same CSV, same slice, same normalisation, same order), so the
#     ONLY acceptable result is bit-identical - tol = 0.
#   * Mackey-Glass: the two generators are INDEPENDENT implementations of the
#     same recursion (the classical track's pre-allocated-array version vs the
#     quantum track's verbatim-from-the-notebook one), evaluated partly in
#     PennyLane's numpy wrapper, so they differ by floating-point round-off
#     (~1e-15) even when mathematically identical. tol = 1e-12 is ~3 orders of
#     magnitude above observed round-off and ~14 orders BELOW the divergence
#     the real 2026-08-30 indexing bug produced (max|diff| 0.58-0.76), so it
#     still catches any genuine drift.
TOL_IDENTICAL = 0.0
TOL_ROUNDOFF = 1e-12


def _report(name, a, b, tol):
    """Compare two series; return True if they agree within `tol`."""
    if len(a) != len(b):
        print(f"  FAIL {name}: different lengths ({len(a)} vs {len(b)})")
        return False
    max_abs = float(np.max(np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))))
    passed = max_abs <= tol
    bar = "bit-identical" if tol == 0.0 else f"tol {tol:.0e}"
    print(f"  {'OK  ' if passed else 'FAIL'} {name}: n={len(a)}  max|diff| = {max_abs:.3e}  "
          f"[{bar}]  (range [{np.min(a):.4f}, {np.max(a):.4f}])")
    return passed


def main():
    ok = True

    print("Real-world series (classical datasets_b vs quantum datasets_qml):")
    for name in REALWORLD:
        classical = datasets_b.DATASETS[name](n_samples=200)
        quantum = datasets_qml.load_realworld_series(name, n_samples=200)
        ok &= _report(name, classical, quantum, TOL_IDENTICAL)

    print("\nMackey-Glass (classical datasets_b vs the quantum track's own "
          "verbatim-from-the-notebook generator):")
    if not HAVE_PENNYLANE:
        print(f"  SKIP - could not import train_monolith_qae ({type(_import_error).__name__}: "
              f"{_import_error}).\n       Install the pinned PennyLane stack "
              f"(../pennylane_monolith/requirements.txt) to include this check.")
    else:
        for tau in TAUS:
            classical = datasets_b.series_mackey_glass(tau=tau)
            # exactly the lines run_one() executes for dataset='mackey_glass'
            raw = qml_train.mackey_glass(length=1200, beta=0.17, gamma=0.1, n=10,
                                          tau=tau, x0=1.2)
            quantum = qml_train.x2y(raw, xlim=(np.min(raw), np.max(raw)), ylim=(0, 1))[3::6]
            ok &= _report(f"mackey_glass tau={tau}", classical, quantum, TOL_ROUNDOFF)

    print()
    if ok:
        print("All checked series are identical across tracks - cross-track "
              "Delta_test values are comparable.")
        return 0
    print("MISMATCH - do NOT put these tracks' numbers in the same table until "
          "the loaders agree.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
