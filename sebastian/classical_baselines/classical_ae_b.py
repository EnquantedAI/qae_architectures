"""
Classical baselines for "Track B" (QUANTICS 2026 extension). Two purposes,
both from the QUANTICS 2026 camera-ready review feedback:

  - TinyMLPAutoencoder: a classical AE deliberately PARAMETER-MATCHED to the
    QAE configs (Reviewer 4: "a small classical denoising autoencoder, sized
    to match the parameter count of the QAE configurations, would ... let
    the reader calibrate the reported 67% noise removal at low
    implementation cost"). QUANTICS 2026's own Fig. 9 finds the Monolith QAE
    saturates at ~140 trainable params for their headline config (window=6,
    latent=4). size_for_param_budget(6, 4, 140) -> hidden=6 -> 142 params,
    about as close as an integer hidden size gets.

  - MLPAutoencoder / LSTMAutoencoder: reused from Track A (project/), same
    architectures, just re-sized for window=6 here - the "generous budget"
    classical comparison (~thousands of params instead of ~140).

The other reviewer-driven change lives in the training loop, not the
models: QUANTICS 2026 resamples the additive noise FRESH every training
iteration (not a static noisy dataset), sigma=0.2, clipped to the series'
value range - see `add_dynamic_noise()` and `train_dynamic_noise()`.
"""

import numpy as np
import torch
import torch.nn as nn

from datasets_b import make_windows  # noqa: F401 (used below)


# ---------------------------------------------------------------------------
# Architectures
# ---------------------------------------------------------------------------

class TinyMLPAutoencoder(nn.Module):
    """Single hidden layer encoder/decoder, parameter-matched to the QAE
    (see module docstring). Default window=6/latent=4/hidden=6 -> 142 params,
    QUANTICS 2026's own headline config (Table 2)."""

    def __init__(self, window_size=6, latent_dim=4, hidden=6):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(window_size, hidden), nn.ReLU(),
            nn.Linear(hidden, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, window_size),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


def size_for_param_budget(window_size, latent_dim, target_params):
    """Hidden size h (for TinyMLPAutoencoder) whose param count is closest
    to `target_params`. total(h) = 2*h*(window+latent+1) + (window+latent)."""
    best_h, best_diff = 1, None
    for h in range(1, 200):
        total = 2 * h * (window_size + latent_dim + 1) + (window_size + latent_dim)
        diff = abs(total - target_params)
        if best_diff is None or diff < best_diff:
            best_h, best_diff = h, diff
    return best_h


class MLPAutoencoder(nn.Module):
    """3-hidden-layer encoder + decoder, the "generous budget" classical
    baseline (thousands of params, not parameter-matched)."""

    def __init__(self, window_size=6, latent_dim=4, hidden=(88, 66, 44)):
        super().__init__()
        h1, h2, h3 = hidden
        self.encoder = nn.Sequential(
            nn.Linear(window_size, h1), nn.ReLU(),
            nn.Linear(h1, h2), nn.ReLU(),
            nn.Linear(h2, h3), nn.ReLU(),
            nn.Linear(h3, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, h3), nn.ReLU(),
            nn.Linear(h3, h2), nn.ReLU(),
            nn.Linear(h2, h1), nn.ReLU(),
            nn.Linear(h1, window_size),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


class LSTMAutoencoder(nn.Module):
    """Sequence-to-sequence LSTM autoencoder (same design as Track A)."""

    def __init__(self, window_size=6, latent_dim=4, hidden_size=16):
        super().__init__()
        self.window_size = window_size
        self.latent_dim = latent_dim
        self.hidden_size = hidden_size
        self.enc_lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, batch_first=True)
        self.to_latent = nn.Linear(hidden_size, latent_dim)
        self.from_latent = nn.Linear(latent_dim, hidden_size)
        self.dec_lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, batch_first=True)
        self.out_proj = nn.Linear(hidden_size, 1)

    def forward(self, x):
        b = x.shape[0]
        seq = x.unsqueeze(-1)
        _, (h_n, c_n) = self.enc_lstm(seq)
        latent = self.to_latent(h_n[-1])
        h0 = self.from_latent(latent).unsqueeze(0)
        c0 = torch.zeros_like(h0)
        dec_input = torch.zeros(b, 1, 1, device=x.device)
        outputs = []
        h, c = h0, c0
        for _ in range(self.window_size):
            out, (h, c) = self.dec_lstm(dec_input, (h, c))
            step = self.out_proj(out)
            outputs.append(step)
            dec_input = step
        return torch.cat(outputs, dim=1).squeeze(-1)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def default_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Dynamic-noise training - CORRECTED 2026-08-29 against the real QUANTICS
# 2026 source (`qae_architectures/qae_utils/Tools.py::ts_add_noise`, and the
# training loop in e.g. `Jacob/4q_2l_2t_tau17.ipynb`, function
# `train_stage3_series_noise_simple`). Two things were wrong in the original
# version of this module (guessed from the PDF alone, before the real code
# was available):
#
#   1. Noise must be added to the WHOLE clean series ONCE per epoch, THEN
#      windowed - not sampled independently per window/per batch element.
#      Real code: `make_windows_from_series()` calls `ts_add_noise(y_full, ...)`
#      on the full series, then `ts_wind_make()` slices it into overlapping
#      windows. Because windows overlap (stride < window), independent
#      per-window noise breaks the correlation the real noise process has
#      between adjacent windows - and, once overlap-averaged for scoring,
#      that mismatch was artificially deflating the "noisy" baseline's own
#      error, which is what produced the negative Delta_test seen in the
#      first smoke test.
#   2. Real code uses `clip=False` at every call site, and scales noise by
#      `sigma * (range_high - range_low)` of the series (real code passes
#      the series' own `scale_low`/`scale_high`, i.e. its min/max) - NOT a
#      hardcoded clip to [0,1]. (Note: this actually contradicts the QUANTICS
#      2026 paper's own Sec. 4 prose, which says noise is "clipped to the
#      maximum range of the time series values" - the released code and the
#      paper's text disagree; we follow the code, since that's what actually
#      produced their published Table 2 numbers.) Because all series here are
#      already normalised to [0,1] (range=1.0), sigma is still directly the
#      noise standard deviation in the same units as before - only the
#      clipping behaviour and the "whole-series-first" ordering change.
#
# Training noise is re-sampled with a NEW seed every epoch (real code:
# `TRAIN_SEED_BASE + ep`), confirming the paper's "resampled dynamically
# during each training iteration" description IS accurate for the noise
# schedule (just not for clipping). Test/eval noise uses one FIXED seed
# (real code: `TEST_SEED_FIXED=99123` - the exact seed found in the real
# eval CSVs `Jacob/qae_eval_framework/eval_all_*.csv`), so results are
# reproducible.
# ---------------------------------------------------------------------------

TEST_SEED_FIXED = 99123  # matches the real repo's eval convention exactly


def add_noise_to_series(series, sigma, seed, lo=None, hi=None, clip=False, noise_type="normal"):
    """Matches qae_utils/Tools.py::ts_add_noise exactly (normal noise,
    scaled by sigma*(hi-lo) of the series' own range, clip=False by
    default)."""
    series = np.asarray(series, dtype=float)
    if lo is None:
        lo = series.min()
    if hi is None:
        hi = series.max()
    rng = np.random.default_rng(seed)
    if noise_type == "normal":
        noise_vec = rng.normal(0, 1, len(series))
    elif noise_type == "uniform":
        noise_vec = rng.uniform(-1, 1, len(series))
    else:
        noise_vec = np.zeros(len(series))
    noisy = series + noise_vec * sigma * (hi - lo)
    if clip:
        noisy = np.clip(noisy, lo, hi)
    return noisy


def train_dynamic_noise(model, series, sigma, window, stride, train_frac=0.75, epochs=300,
                         lr=2e-3, batch_size=8, seed=2023, device=None, log_every=0,
                         test_seed_fixed=TEST_SEED_FIXED):
    """Train a denoising AE where the WHOLE series is re-noised every epoch
    (fresh seed each epoch), then windowed - matching QUANTICS 2026's real
    training loop. Test noise uses one fixed seed throughout, also matching
    the real code."""
    device = device or default_device()
    torch.manual_seed(seed)
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    series = np.asarray(series, dtype=float)
    clean_windows = make_windows(series, window, stride)
    n_train = int(round(len(clean_windows) * train_frac))
    train_clean_w = clean_windows[:n_train]
    test_clean_w = clean_windows[n_train:]
    Ytr = torch.tensor(train_clean_w, dtype=torch.float32, device=device)
    Yte = torch.tensor(test_clean_w, dtype=torch.float32, device=device)
    n = Ytr.shape[0]

    # fixed test noise: whole series noised once with test_seed_fixed, then windowed
    noisy_series_test = add_noise_to_series(series, sigma, seed=test_seed_fixed)
    Xte = torch.tensor(make_windows(noisy_series_test, window, stride)[n_train:],
                        dtype=torch.float32, device=device)

    train_losses, test_losses = [], []
    for epoch in range(epochs):
        # epoch-specific train noise: whole series, fresh seed each epoch
        noisy_series_ep = add_noise_to_series(series, sigma, seed=seed * 100_000 + epoch)
        Xtr = torch.tensor(make_windows(noisy_series_ep, window, stride)[:n_train],
                            dtype=torch.float32, device=device)

        model.train()
        perm = torch.randperm(n, device=device)
        epoch_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = Xtr[idx], Ytr[idx]
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * len(idx)
        train_losses.append(epoch_loss / n)

        model.eval()
        with torch.no_grad():
            vpred = model(Xte)
            vloss = loss_fn(vpred, Yte).item()
        test_losses.append(vloss)

        if log_every and (epoch % log_every == 0 or epoch == epochs - 1):
            print(f"  epoch {epoch+1}/{epochs}  train_mse={train_losses[-1]:.5f}  test_mse={vloss:.5f}")

    return {"train_losses": train_losses, "test_losses": test_losses}


def eval_fixed_noise(model, series, sigma, window, stride, train_frac=0.75,
                      seed=TEST_SEED_FIXED, device=None):
    """One FIXED noisy realisation of the whole series (same seed convention
    as training's test noise / the real eval CSVs' seed=99123), windowed and
    split the same way as training, for final R2/MAE/RMSE reporting."""
    device = device or next(model.parameters()).device
    series = np.asarray(series, dtype=float)
    clean_windows = make_windows(series, window, stride)
    n_train = int(round(len(clean_windows) * train_frac))
    test_clean_w = clean_windows[n_train:]

    noisy_series = add_noise_to_series(series, sigma, seed=seed)
    noisy_windows = make_windows(noisy_series, window, stride)
    test_noisy_w = noisy_windows[n_train:]

    model.eval()
    with torch.no_grad():
        x = torch.tensor(test_noisy_w, dtype=torch.float32, device=device)
        rec = model(x).cpu().numpy()
    return test_noisy_w, rec, test_clean_w


def to_window_dict(arr):
    return {i: list(arr[i]) for i in range(len(arr))}
