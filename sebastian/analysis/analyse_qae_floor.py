"""
Fit an empirical affine model of QAE reconstruction error versus noise power.

The intercept is a regression coefficient, not the measured zero-noise error
or an established irreducible architectural limit. Models are trained
separately at each sigma. Break-even estimates describe the recorded protocol.
PCA residual variance is an exploratory linear-compressibility descriptor;
a four-qubit register is not equivalent to four classical PCA coordinates.

Run from the package root: python analysis/analyse_qae_floor.py
"""

import json
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
QML_ROWS = os.path.join(ROOT, "pennylane_monolith", "results_qml", "monolith_qml.json")

# MSE of the fixed evaluation noise realisation at sigma=0.2, used to convert a
# break-even MSE back into a sigma (noise power scales as sigma^2).
MSE_NOISE_AT_02 = 0.05266095


def label(row):
    ds = row.get("dataset", "mackey_glass")
    return f"MG tau={row['tau']}" if ds == "mackey_glass" else ds


def fit_floor(points):
    """Least squares on MSE_rec = floor + k * MSE_noise."""
    n = len(points)
    if n < 2:
        return None
    sx = sum(p[0] for p in points)
    sy = sum(p[1] for p in points)
    sxx = sum(p[0] ** 2 for p in points)
    sxy = sum(p[0] * p[1] for p in points)
    denom = n * sxx - sx * sx
    if denom == 0:
        return None
    k = (n * sxy - sx * sy) / denom
    floor = (sy - k * sx) / n
    return floor, k


def main():
    if not os.path.exists(QML_ROWS):
        sys.exit(f"missing {QML_ROWS}")
    rows = json.load(open(QML_ROWS))
    if not rows:
        sys.exit("no rows")

    # canonical epoch count only - never pool smoke-test runs (see
    # aggregate_comparison.py's collect() for why this matters)
    max_epochs = {}
    for r in rows:
        max_epochs[label(r)] = max(max_epochs.get(label(r), 0), r["n_epochs"])
    rows = [r for r in rows if r["n_epochs"] == max_epochs[label(r)]]

    groups = {}
    for r in rows:
        groups.setdefault((label(r), r["sigma"]), []).append(r)

    print("QAE test error vs noise level (mean over seeds)\n")
    print(f"{'series':12s} {'sigma':>5s} {'n':>3s} {'MSE_noise':>10s} {'MSE_rec':>10s} {'Delta%':>9s}")
    per_series = {}
    for key in sorted(groups, key=lambda k: (k[0], k[1])):
        lab, sigma = key
        rs = groups[key]
        mse_noise = rs[0]["mse_test_noise"]
        mse_rec = st.mean(r["mse_test_recovered"] for r in rs)
        delta = st.mean(r["delta_test_pct"] for r in rs) if sigma > 0 else None
        delta_text = f"{delta:+9.2f}" if delta is not None else "undefined"
        per_series.setdefault(lab, []).append((mse_noise, mse_rec))
        print(f"{lab:12s} {sigma:5.2f} {len(rs):3d} {mse_noise:10.5f} "
              f"{mse_rec:10.5f} {delta_text:>9s}")

    print("\nFit  MSE_rec = floor + k * MSE_noise   (floor = fitted intercept; not the zero-noise measurement)\n")
    print(f"{'series':12s} {'floor':>9s} {'k':>7s} {'break-even sigma':>18s}")
    for lab, pts in sorted(per_series.items()):
        fit = fit_floor(pts)
        if fit is None:
            print(f"{lab:12s} {'--':>9s} {'--':>7s} {'need >=2 noise levels':>18s}")
            continue
        floor, k = fit
        if k >= 1:
            print(f"{lab:12s} {floor:9.5f} {k:7.3f} {'n/a (k>=1)':>18s}")
            continue
        mse_cross = floor / (1 - k)
        sigma_cross = 0.2 * (mse_cross / MSE_NOISE_AT_02) ** 0.5
        print(f"{lab:12s} {floor:9.5f} {k:7.3f} {sigma_cross:18.3f}")

    print("\nBreak-even values are empirical estimates for this training/evaluation\n"
          "protocol, conditional on an adequate affine fit and k < 1. They do\n"
          "not prove a training-independent limitation. Delta is undefined at sigma=0.")
    print("\nDirect zero-noise reconstruction error versus fitted intercept:")
    for lab, pts in sorted(per_series.items()):
        clean = [y for x, y in pts if x == 0]
        fit = fit_floor(pts)
        if clean and fit is not None:
            print(f"{lab:12s} clean MSE={clean[0]:.8f}  fitted intercept={fit[0]:.8f}")

    compressibility(per_series)


def compressibility(per_series):
    """Compare linear PCA compressibility with fitted intercepts.

    This is exploratory: four series, one with substantial leverage. Four
    principal components do not represent a four-qubit information bound.
    """
    try:
        import numpy as np
        sys.path.insert(0, os.path.join(ROOT, "classical_baselines"))
        import datasets_b as db
    except Exception as exc:
        print(f"\n(skipping compressibility check: {type(exc).__name__}: {exc})")
        return

    name_map = {"MG tau=17": "mackey_glass_tau17", "MG tau=30": "mackey_glass_tau30",
                "energy": "energy", "finance": "finance"}
    print("\n\nWindow compressibility (PCA of the clean windows, W=6, 4 principal components)\n")
    print(f"{'series':12s} {'var kept by 4 PCs':>18s} {'residual':>10s} {'fitted floor':>13s}")
    resid, floors = [], []
    for lab in sorted(per_series):
        ds = name_map.get(lab)
        if ds is None:
            continue
        fit = fit_floor(per_series[lab])
        if fit is None:
            continue
        _, _, series = db.prepare_dataset(ds, window_size=6, stride=2)
        W = db.make_windows(np.asarray(series), 6, 2)
        ev = np.linalg.svd(W - W.mean(0), compute_uv=False) ** 2
        ev = ev / ev.sum()
        keep = float(ev[:4].sum())
        resid.append(1 - keep)
        floors.append(fit[0])
        print(f"{lab:12s} {keep * 100:17.3f}% {(1 - keep) * 100:9.3f}% {fit[0]:13.5f}")
    if len(resid) > 2:
        r = float(np.corrcoef(resid, floors)[0, 1])
        print(f"\nPearson r(residual variance, fitted floor) = {r:.4f}  (n={len(resid)} series)")
        print("Caveat: n is tiny and one series (energy) dominates the spread, so treat\n"
              "this as an exploratory association, not as a validated predictor or a law.")


if __name__ == "__main__":
    main()
