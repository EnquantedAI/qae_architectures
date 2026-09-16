"""
Real-world series loaders for the quantum track, byte-identical to the
classical track's own loaders (`classical_baselines/datasets_b.py`:
`series_energy`, `series_finance`).

WHY THIS EXISTS
---------------
QUANTICS 2026 ran the QAE on Mackey-Glass only. The CMES extension needs the
same Monolith QAE run on at least one NON-CHAOTIC, real-world series, so the
paper's classical-vs-quantum comparison is not a single-dataset claim. The
classical side already has energy/finance under the QUANTICS 2026 protocol
(classical_baselines/results_b/classical_grid_b.json); this module is the
missing quantum-side half.

THE ONE RULE THIS MODULE EXISTS TO ENFORCE
------------------------------------------
The two tracks must see NUMERICALLY IDENTICAL series - otherwise their
Delta_test numbers cannot be put in the same table (exactly the class of bug
the 2026-08-30 audit found for Mackey-Glass, where a different generator made
the two tracks silently non-comparable). The loaders below are therefore a
deliberate, line-for-line mirror of datasets_b.py's, including the ORDER of
operations:

    slice the first n_samples points  ->  THEN min-max normalise to [0,1]

(not normalise-then-slice, which gives different values). `verify_tracks_match.py`
asserts equality to 0.0 max-abs-difference and should be run after any change
to either module.

Mackey-Glass is deliberately NOT here: it stays in train_monolith_qae.py's own
verbatim-from-the-real-notebook `mackey_glass()` function, so the 30 already-
computed MG rows in results_qml/monolith_qml.json remain reproducible
bit-for-bit.
"""

import os

import numpy as np
import pandas as pd

# Data lives in the classical package (shared, not duplicated per-track).
# Search order: explicit env var -> this package's own data/ -> the classical
# package's data/ (the usual layout when both are unzipped side by side).
_CANDIDATE_DATA_DIRS = [
    os.environ.get("QAE_DATA_DIR", ""),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                 "classical_baselines", "data"),
]


def _data_path(filename):
    tried = []
    for d in _CANDIDATE_DATA_DIRS:
        if not d:
            continue
        p = os.path.join(d, filename)
        tried.append(p)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f"Could not find {filename}. Looked in:\n  " + "\n  ".join(tried) +
        "\nSet QAE_DATA_DIR to the directory holding the CSVs, or copy "
        "classical_baselines/data/ next to this script.")


def _normalise_to_01(values):
    """Identical to datasets_b.py::_normalise_to_01."""
    values = np.asarray(values, dtype=float)
    lo, hi = values.min(), values.max()
    return (values - lo) / (hi - lo)


def series_energy(n_samples=200):
    """Victoria (AU) grid electricity demand, half-hourly -> daily totals,
    first `n_samples` days, min-max normalised to [0,1].

    Mirrors datasets_b.py::series_energy exactly."""
    df = pd.read_csv(_data_path("energy_vic_elec.csv"))
    df["date"] = pd.to_datetime(df["Date"], unit="D", origin="1899-12-30")
    daily = df.groupby("date")["OperationalLessIndustrial"].sum().sort_index()
    return _normalise_to_01(daily.to_numpy()[:n_samples])


def series_finance(n_samples=200):
    """AAPL daily close, first `n_samples` trading days, min-max normalised
    to [0,1].

    Mirrors datasets_b.py::series_finance exactly."""
    df = pd.read_csv(_data_path("finance_aapl.csv"))
    return _normalise_to_01(df["AAPL.Close"].to_numpy()[:n_samples])


# Real-world (non-chaotic) series only. "mackey_glass" is handled inside
# train_monolith_qae.py by its own verbatim generator - see module docstring.
REALWORLD_LOADERS = {
    "energy": series_energy,
    "finance": series_finance,
}

DATASET_CHOICES = ["mackey_glass"] + sorted(REALWORLD_LOADERS)


def load_realworld_series(name, n_samples=200):
    if name not in REALWORLD_LOADERS:
        raise KeyError(f"unknown real-world dataset {name!r}; "
                       f"expected one of {sorted(REALWORLD_LOADERS)}")
    series = REALWORLD_LOADERS[name](n_samples=n_samples)
    assert len(series) == n_samples, \
        f"{name}: expected {n_samples} points, got {len(series)}"
    assert np.isclose(series.min(), 0.0) and np.isclose(series.max(), 1.0), \
        f"{name}: expected a [0,1]-normalised series, got " \
        f"[{series.min():.4f}, {series.max():.4f}]"
    return series
