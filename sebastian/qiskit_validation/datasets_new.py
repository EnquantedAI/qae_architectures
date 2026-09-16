"""
New MVP datasets for the CMES extension:

- Target_finance: AAPL daily closing price (real, plotly/datasets mirror of
  Yahoo Finance data, ~506 trading days, 2015-2017). Strongly stochastic,
  close to a random walk - a useful contrast to the beer sales' seasonality.
  Non-chaotic real-world data.
- Target_energy: Victoria (Australia) grid electricity demand, aggregated to
  daily totals (real, tidyverts/tsibbledata mirror of AEMO grid operator
  data). Strong weekly/annual seasonality + measurement noise - and links
  thematically to Hyndman & Athanasopoulos [14], already cited in the paper.
  Non-chaotic real-world data.
- Target_mackey_glass: synthetic Mackey-Glass delay-differential system,
  the standard chaos benchmark in the nonlinear-dynamics / time-series-ML
  literature. tau=17 (default) is the classic mildly-chaotic parametrisation
  (the system is chaotic for tau > ~16.8; larger tau => more chaotic). This
  is a deliberate CONTRAST case to the three non-chaotic datasets above -
  useful to show QuTSAE's denoising behaviour generalises across both
  chaotic and non-chaotic dynamics, not just the non-chaotic regime.

All three follow the same `Target` interface as utils/Target.py's
Target_beer, so they drop straight into utils.TS.gen_ts_windows() unchanged.
"""

import os

import numpy as np
import pandas as pd

from utils.Target import Target

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def _normalise_to_01(values):
    values = np.asarray(values, dtype=float)
    lo, hi = values.min(), values.max()
    return (values - lo) / (hi - lo)


def generate_mackey_glass(n_samples, tau=17, beta=0.2, gamma=0.1, n=10,
                           dt=1.0, x0=0.9, burn_in=1000, seed=42):
    """Discrete-time Mackey-Glass series (Euler integration, dt=1), the same
    recursion used throughout the nonlinear-dynamics/ML-benchmark literature
    and in this repo's book/part3/ch09_qae_pure.qmd teaching chapter:

        dx/dt = beta * x(t-tau) / (1 + x(t-tau)**n) - gamma * x(t)

    tau=17 (with the typical beta=0.2, gamma=0.1, n=10) is the standard
    "onset of chaos" benchmark parametrisation (chaotic for tau > ~16.8).

    Unlike the book chapter's minimal version, this adds a `burn_in` period:
    the recursion starts from a constant history (x=x0 for t in [-tau, 0]),
    which is NOT on the chaotic attractor yet, so the first ~hundreds of
    steps are a transient. We integrate burn_in extra steps and discard them
    before returning n_samples, so the returned series is actually sampled
    from the attractor - standard practice, and more defensible for a
    reviewer than using the raw transient.

    `seed` is accepted for interface symmetry with other generators but the
    recursion itself is fully deterministic (no stochastic term) - it only
    seeds numpy's global RNG so downstream noise injection (added later, in
    the shared gen_ts_windows pipeline) is reproducible.
    """
    np.random.seed(seed)
    total = n_samples + burn_in
    x = [x0] * (tau + 1)
    for _ in range(total + tau):
        x_tau = x[-tau] if tau > 0 else x[-1]
        dx = beta * x_tau / (1 + x_tau ** n) - gamma * x[-1]
        x.append(x[-1] + dt * dx)
    x = np.array(x[tau + 1:])       # drop the constant seed history
    x = x[burn_in:burn_in + n_samples]  # drop the transient before the attractor
    return x


class Target_mackey_glass(Target):
    """Synthetic Mackey-Glass chaotic benchmark, normalised to [0, 1] (same
    convention as Target_beer). tau=17 => classic mildly-chaotic regime."""

    name = "Target_mackey_glass"

    def __init__(self, pt_from=None, pt_to=None, tau=17, n_samples=500,
                 burn_in=1000, seed=42):
        super().__init__()
        self.tau = tau
        raw = generate_mackey_glass(n_samples, tau=tau, burn_in=burn_in, seed=seed)
        pt_from = 0 if pt_from is None else pt_from
        pt_to = len(raw) if pt_to is None else pt_to
        self.ts_data = list(_normalise_to_01(raw)[pt_from:pt_to])
        self.ts_len = len(self.ts_data)
        self.xmin = 0
        self.xmax = self.ts_len - 1
        self.ymin = min(self.ts_data)
        self.ymax = max(self.ts_data)
        self.epsilon = 0.1

    def fun_point(self, x):
        if x < self.xmin or x > self.xmax:
            return 0.0
        if int(x) == self.xmax:
            return self.ts_data[-1]
        lx = int(x)
        ux = lx + 1
        ly, uy = self.ts_data[lx], self.ts_data[ux]
        return ly + (x - lx) * (uy - ly) / (ux - lx)

    def fun(self, x):
        if isinstance(x, (int, float, np.floating)):
            return self.fun_point(x)
        return np.array([self.fun_point(xi) for xi in x])


class Target_finance(Target):
    """AAPL daily closing price, normalised to [0, 1] (same convention as
    Target_beer). Source: plotly/datasets `finance-charts-apple.csv`
    (mirrors Yahoo Finance AAPL OHLC data, ~2015-02 to 2017-02)."""

    name = "Target_finance"

    def __init__(self, pt_from=None, pt_to=None, csv_path=None):
        super().__init__()
        csv_path = csv_path or os.path.join(DATA_DIR, "finance_aapl.csv")
        df = pd.read_csv(csv_path)
        closes = df["AAPL.Close"].to_numpy()
        pt_from = 0 if pt_from is None else pt_from
        pt_to = len(closes) if pt_to is None else pt_to
        self.ts_data = list(_normalise_to_01(closes[pt_from:pt_to]))
        self.ts_len = len(self.ts_data)
        self.xmin = 0
        self.xmax = self.ts_len - 1
        self.ymin = min(self.ts_data)
        self.ymax = max(self.ts_data)
        self.epsilon = 0.1

    def fun_point(self, x):
        if x < self.xmin or x > self.xmax:
            return 0.0
        if int(x) == self.xmax:
            return self.ts_data[-1]
        lx = int(x)
        ux = lx + 1
        ly, uy = self.ts_data[lx], self.ts_data[ux]
        return ly + (x - lx) * (uy - ly) / (ux - lx)

    def fun(self, x):
        if isinstance(x, (int, float, np.floating)):
            return self.fun_point(x)
        return np.array([self.fun_point(xi) for xi in x])


class Target_energy(Target):
    """Victoria (AU) grid electricity demand, aggregated from half-hourly to
    daily totals, normalised to [0, 1]. Source: tidyverts/tsibbledata
    VIC2015/demand.csv (real AEMO grid-operator data)."""

    name = "Target_energy"

    def __init__(self, pt_from=None, pt_to=None, csv_path=None, n_days=None):
        super().__init__()
        csv_path = csv_path or os.path.join(DATA_DIR, "energy_vic_elec.csv")
        df = pd.read_csv(csv_path)
        # Date is an Excel serial date (origin 1899-12-30); Period is the
        # half-hour-of-day index (1..48). Aggregate to daily totals.
        df["date"] = pd.to_datetime(df["Date"], unit="D", origin="1899-12-30")
        daily = df.groupby("date")["OperationalLessIndustrial"].sum()
        daily = daily.sort_index()
        values = daily.to_numpy()
        if n_days is not None:
            values = values[:n_days]
        pt_from = 0 if pt_from is None else pt_from
        pt_to = len(values) if pt_to is None else pt_to
        self.ts_data = list(_normalise_to_01(values[pt_from:pt_to]))
        self.ts_len = len(self.ts_data)
        self.xmin = 0
        self.xmax = self.ts_len - 1
        self.ymin = min(self.ts_data)
        self.ymax = max(self.ts_data)
        self.epsilon = 0.1

    def fun_point(self, x):
        if x < self.xmin or x > self.xmax:
            return 0.0
        if int(x) == self.xmax:
            return self.ts_data[-1]
        lx = int(x)
        ux = lx + 1
        ly, uy = self.ts_data[lx], self.ts_data[ux]
        return ly + (x - lx) * (uy - ly) / (ux - lx)

    def fun(self, x):
        if isinstance(x, (int, float, np.floating)):
            return self.fun_point(x)
        return np.array([self.fun_point(xi) for xi in x])
