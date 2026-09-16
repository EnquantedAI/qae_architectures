"""
Data generation and windowing for "Track B" - matching the QUANTICS 2026
paper's exact experimental setup (Section 4 "Experimental design"), NOT
paper_279/ICCS-2024's convention (that's project/, "Track A").

Key differences from Track A that this module implements faithfully:
  - Mackey-Glass generated with their exact recipe: Forward Euler, dt=1,
    1200 raw steps, downsampled by factor 6 -> 200 values. NO burn-in
    discard (unlike Track A's Target_mackey_glass) - matching their
    described procedure exactly.
  - Two tau values: 17 (mentioned in their Section 4 but never revisited in
    their Results - this is what CR Reviewer 4 asked to have reintroduced)
    and 30 (their actual headline/reported results, Table 2: 67.15% at
    window=6).
  - Windowing: window size W in {4, 6} (we focus on W=6, their "scalability"
    config used for Table 2's headline numbers - qubits = W), stride = 2
    (not paper_279's stride=4).
  - Chronological 75/25 train/test split (first 75% of windows = train, no
    shuffling of window order). For W=6 on a 200-sample series this gives 98
    windows total (74 train / 24 test), matching their reported counts
    exactly. NOTE (corrected 2026-08-30): the real repo's own code comment
    calls this "avoid data leakage", but that's not fully accurate and we
    don't repeat the claim here - two real leakage paths remain, INHERITED
    from the real methodology (not something this module fixes, to stay
    faithful to what actually produced the published numbers): (a) since
    stride(2) < window(6), the last train window and first test window
    still share raw sample points at the boundary; (b) `_normalise_to_01`
    below uses the min/max of the WHOLE series, including the test portion,
    so test-set values leak into the normalisation of training data. Both
    should be disclosed as known limitations in the paper, not silently
    presented as leakage-free.
  - Noise: NOT a static pre-computed noisy dataset. Sigma=0.2 Gaussian noise
    is resampled fresh every training iteration/epoch and clipped to the
    min/max range of the clean series (their exact wording: "noise was
    resampled dynamically during each training iteration ... and clipped to
    the maximum range of the time series values"). See
    `add_dynamic_noise()` below, used inside classical_ae_b.py's training
    loop (NOT here - this module only prepares the clean windowed data).

Beyond their own scope (Mackey-Glass only), we add energy/finance (real,
non-chaotic) and beer (paper_279's own dataset, for continuity) run through
the SAME windowing/noise convention above, so the whole Track B comparison
(classical vs. whatever quantum Track B produces) is internally consistent.
"""

import os

import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


# ---------------------------------------------------------------------------
# Mackey-Glass, exact QUANTICS 2026 recipe
# ---------------------------------------------------------------------------

def mackey_glass_quantics(tau, length=1200, beta=0.17, gamma=0.1, n=10, x0=1.2):
    """Forward-Euler Mackey-Glass recursion, matching the real QUANTICS 2026
    training notebook's own `mackey_glass()` function EXACTLY (verified
    numerically against `project_b_qml/train_monolith_qae.py::mackey_glass`,
    which was extracted near-verbatim from the real notebook):

        dx/dt = beta*x(t-tau) / (1 + x(t-tau)**n) - gamma*x(t)   [theta=1]

    FIXED 2026-08-30: this used to be a growing-Python-list implementation
    with NEGATIVE indexing (`x[-tau]`), which is off-by-one and produces a
    genuinely different (incorrect) chaotic series from index=tau onward -
    a real bug found while auditing this code, independent of the
    beta/downsample issues below (max abs diff vs. the correct series:
    0.58 for tau=17, 0.76 for tau=30 - not a rounding difference). Replaced
    with the real notebook's own pre-allocated-array/absolute-indexing
    approach.

    Also FIXED: beta now defaults to 0.17 (the real notebook's own value -
    NOT the QUANTICS 2026 paper text's stated 0.2, which the paper and its
    own released code disagree on; we follow the code, since that's what
    produced the published Table 2 numbers). length=1200 raw steps, no
    burn-in (matches the real notebook). Downsampling (offset=3, not 0) now
    happens in `series_mackey_glass` below, AFTER normalisation, matching
    the real notebook's own order of operations.
    """
    x = np.zeros(length + tau + 1, dtype=float)
    x[:tau + 1] = x0
    for t in range(tau, length + tau):
        xt = x[t]
        xt_tau = x[t - tau]
        dx = beta * xt_tau / (1.0 + xt_tau ** n) - gamma * xt
        x[t + 1] = xt + dx
    return x[tau + 1:]


def _normalise_to_01(values):
    values = np.asarray(values, dtype=float)
    lo, hi = values.min(), values.max()
    return (values - lo) / (hi - lo)


# ---------------------------------------------------------------------------
# Series loaders - all return a length-N, [0,1]-normalised 1D np.array
# ---------------------------------------------------------------------------

def series_mackey_glass(tau, n_samples=200, seed=None):
    raw = mackey_glass_quantics(tau=tau)  # length=1200 raw points, beta=0.17
    normalised = _normalise_to_01(raw)  # min/max over the FULL 1200-pt series,
                                         # matching the real notebook's order
                                         # (normalise, THEN downsample)
    downsampled = normalised[3::6]  # FIXED 2026-08-30: offset=3, matches the
                                     # real notebook's `data_y[3::6]` exactly
                                     # (this used to be `x[::6]`, offset=0)
    assert len(downsampled) == n_samples, f"expected {n_samples} samples, got {len(downsampled)}"
    return downsampled


def series_energy(n_samples=200):
    df = pd.read_csv(os.path.join(DATA_DIR, "energy_vic_elec.csv"))
    df["date"] = pd.to_datetime(df["Date"], unit="D", origin="1899-12-30")
    daily = df.groupby("date")["OperationalLessIndustrial"].sum().sort_index()
    return _normalise_to_01(daily.to_numpy()[:n_samples])


def series_finance(n_samples=200):
    df = pd.read_csv(os.path.join(DATA_DIR, "finance_aapl.csv"))
    return _normalise_to_01(df["AAPL.Close"].to_numpy()[:n_samples])


BEER_DATA = [
    0.097, 0.0, 0.033, 0.124, 0.113, 0.042, 0.088, 0.08, 0.078, 0.055,
    0.077, 0.138, 0.148, 0.135, 0.302, 0.165, 0.203, 0.187, 0.203, 0.242,
    0.353, 0.269, 0.281, 0.357, 0.359, 0.376, 0.631, 0.213, 0.275, 0.281,
    0.287, 0.291, 0.288, 0.337, 0.426, 0.382, 0.179, 0.165, 0.174, 0.218,
    0.196, 0.225, 0.22, 0.272, 0.22, 0.273, 0.463, 0.205, 0.274, 0.309,
    0.541, 0.581, 0.41, 0.095, 0.163, 0.194, 0.325, 0.301, 0.234, 0.147,
    0.138, 0.132, 0.192, 0.178, 0.295, 0.173, 0.235, 0.299, 0.244, 0.212,
    0.311, 0.296, 0.531, 0.51, 0.379, 0.447, 0.414, 0.471, 0.776, 0.344,
    0.389, 0.353, 0.366, 0.411, 0.435, 0.393, 0.453, 0.404, 0.327, 0.338,
    0.255, 0.269, 0.217, 0.219, 0.252, 0.278, 0.197, 0.207, 0.337, 0.561,
    0.223, 0.312, 0.53, 0.652, 0.493, 0.131, 0.16, 0.343, 0.264, 0.178,
    0.205, 0.221, 0.222, 0.179, 0.206, 0.237, 0.251, 0.24, 0.293, 0.555,
    0.3, 0.282, 0.332, 0.396, 0.603, 0.515, 0.379, 0.476, 0.433, 0.536,
    1.0, 0.474, 0.459, 0.471, 0.458, 0.448, 0.465, 0.484, 0.65, 0.494,
    0.37, 0.358, 0.313, 0.303, 0.29, 0.245, 0.235, 0.322, 0.208, 0.226,
    0.383, 0.679, 0.231, 0.35, 0.518, 0.806, 0.655, 0.177, 0.238, 0.229,
    0.431, 0.338, 0.228, 0.219, 0.231, 0.246, 0.285, 0.307, 0.253, 0.347,
    0.468, 0.331, 0.383, 0.369, 0.379, 0.481, 0.446, 0.685, 0.585, 0.474,
    0.548, 0.498, 0.907, 0.606, 0.469, 0.462, 0.447, 0.493, 0.51, 0.472,
    0.467, 0.669, 0.591, 0.396, 0.294, 0.342, 0.39, 0.353, 0.359, 0.368,
    0.251, 0.32, 0.419, 0.683, 0.23, 0.36, 0.535, 0.819, 0.752, 0.193,
    0.235, 0.297, 0.259, 0.465, 0.359, 0.209, 0.21, 0.23, 0.264, 0.34,
    0.451, 0.266, 0.293, 0.346, 0.312, 0.299, 0.311, 0.41, 0.414, 0.692,
    0.577, 0.487, 0.545, 0.622, 0.95, 0.782, 0.51, 0.532, 0.566, 0.61,
    0.581, 0.553, 0.558, 0.68, 0.552, 0.35, 0.331, 0.376, 0.434, 0.412,
    0.343, 0.311, 0.335, 0.318, 0.458, 0.821, 0.315, 0.341, 0.487, 0.954,
    0.719, 0.293, 0.282, 0.243, 0.291, 0.576, 0.449, 0.25, 0.267, 0.267,
    0.303, 0.36, 0.279, 0.311, 0.288, 0.425, 0.246, 0.272, 0.297, 0.33,
    0.339, 0.641, 0.562, 0.4, 0.503, 0.506, 0.667, 0.73, 0.411, 0.418,
    0.422, 0.471, 0.468, 0.471, 0.449, 0.567, 0.484, 0.332, 0.292, 0.28,
    0.337, 0.288, 0.275, 0.278, 0.275, 0.294, 0.442, 0.668, 0.179, 0.275,
    0.377, 0.716,
]


def series_beer(n_samples=200):
    # Interpolate the 302-point beer series down to n_samples points, same
    # linspace-resampling convention Track A uses (via gen_ts), so this is
    # comparable in spirit even though beer isn't length-200 natively.
    raw = np.asarray(BEER_DATA)
    xp = np.arange(len(raw))
    x_new = np.linspace(0, len(raw) - 1, n_samples)
    resampled = np.interp(x_new, xp, raw)
    return _normalise_to_01(resampled)


DATASETS = {
    "mackey_glass_tau17": lambda: series_mackey_glass(tau=17),
    "mackey_glass_tau30": lambda: series_mackey_glass(tau=30),
    "energy": series_energy,
    "finance": series_finance,
    "beer": series_beer,
}


# ---------------------------------------------------------------------------
# Windowing (QUANTICS 2026 Sec. 4 "Data preparation")
# ---------------------------------------------------------------------------

def make_windows(series, window_size, stride):
    """Sliding windows, non-differenced, no shuffling (chronological)."""
    n = len(series)
    starts = range(0, n - window_size + 1, stride)
    return np.array([series[s:s + window_size] for s in starts])


def chronological_split(windows, train_frac=0.75):
    n_train = int(round(len(windows) * train_frac))
    return windows[:n_train], windows[n_train:]


def prepare_dataset(name, window_size=6, stride=2, n_samples=200, train_frac=0.75):
    """Returns (train_windows, test_windows), both clean (noise-free),
    shape (n_windows, window_size), values in [0,1]. Noise is added later,
    dynamically, inside the training loop (see classical_ae_b.py)."""
    series = DATASETS[name](n_samples=n_samples) if name in (
        "energy", "finance", "beer") else DATASETS[name]()
    windows = make_windows(series, window_size, stride)
    train, test = chronological_split(windows, train_frac)
    return train, test, series
