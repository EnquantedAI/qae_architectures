"""
Classical autoencoder baselines for the QuTSAE extension (CMES special issue
"Quantum Machine Learning: Methods and Engineering Applications").

Two architectures, both trained the same way as the paper's PyTorch/MLP
baseline (Sec 3.1, Table 1 Ex.3): same windowed/differenced/angle-scaled
data, denoising objective (noisy in -> pure out), full latent-size sweep.

- MLPAutoencoder: extends the paper's single-lat=7 MLP baseline to the full
  lat=1..8 grid, matching "3 hidden layers each" for encoder/decoder.
- LSTMAutoencoder: NEW second classical baseline (MVP item from the CMES
  extension plan) - a sequence-to-sequence recurrent autoencoder, the more
  natural "off-the-shelf" architecture for time series than a flat MLP.

Both report parameter counts explicitly (paper: MLP has 19,741 params at
lat=7, vs. QuTSAE's 64-180 - a ~110-300x gap worth quantifying, per the
extension plan's "parameter-efficiency" analysis).
"""

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Architectures
# ---------------------------------------------------------------------------

class MLPAutoencoder(nn.Module):
    """3-hidden-layer encoder + 3-hidden-layer decoder MLP, matching the
    paper's architecture description (Sec 3.2: "two multilayer perceptrons
    ... featuring 3 hidden layers each"). Hidden sizes are not given in the
    paper (only the total: 19,741 params @ lat=7); we choose a reasonable
    symmetric taper and report the actual count for transparency."""

    def __init__(self, window_size=8, latent_dim=7, hidden=(32, 24, 16)):
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
    """Sequence-to-sequence LSTM autoencoder: an LSTM encoder compresses the
    window into a fixed-size latent vector (final hidden state, projected to
    `latent_dim`); an LSTM decoder reconstructs the window from it,
    step-by-step (teacher-forcing-free, autoregressive on its own output)."""

    def __init__(self, window_size=8, latent_dim=7, hidden_size=16):
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
        # x: (batch, window_size) -> (batch, window_size, 1)
        b = x.shape[0]
        seq = x.unsqueeze(-1)
        _, (h_n, c_n) = self.enc_lstm(seq)
        latent = self.to_latent(h_n[-1])  # (batch, latent_dim)

        h0 = self.from_latent(latent).unsqueeze(0)  # (1, batch, hidden)
        c0 = torch.zeros_like(h0)

        dec_input = torch.zeros(b, 1, 1, device=x.device)
        outputs = []
        h, c = h0, c0
        for _ in range(self.window_size):
            out, (h, c) = self.dec_lstm(dec_input, (h, c))
            step = self.out_proj(out)  # (batch, 1, 1)
            outputs.append(step)
            dec_input = step  # autoregressive
        return torch.cat(outputs, dim=1).squeeze(-1)  # (batch, window_size)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Training / evaluation (denoising: noisy in -> pure out), matching the
# paper's setup: same windowed/differenced/scaled data as QuTSAE, so results
# are directly comparable via the same metrics.
# ---------------------------------------------------------------------------

def default_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def train_classical_ae(model, X_train_noisy, y_train_pure, X_valid_noisy, y_valid_pure,
                        epochs=300, lr=1e-3, batch_size=8, seed=2023, log_every=50,
                        weight_decay=0.0, verbose=True, device=None):
    device = device or default_device()
    torch.manual_seed(seed)
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    Xtr = torch.tensor(np.asarray(X_train_noisy), dtype=torch.float32, device=device)
    Ytr = torch.tensor(np.asarray(y_train_pure), dtype=torch.float32, device=device)
    Xva = torch.tensor(np.asarray(X_valid_noisy), dtype=torch.float32, device=device)
    Yva = torch.tensor(np.asarray(y_valid_pure), dtype=torch.float32, device=device)

    n = Xtr.shape[0]
    train_losses, valid_losses = [], []

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
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
            vpred = model(Xva)
            vloss = loss_fn(vpred, Yva).item()
        valid_losses.append(vloss)

        if verbose and log_every and (epoch % log_every == 0 or epoch == epochs - 1):
            print(f"  epoch {epoch+1}/{epochs}  train_mse={train_losses[-1]:.5f}  valid_mse={vloss:.5f}")

    return {"train_losses": train_losses, "valid_losses": valid_losses}


def reconstruct(model, X_noisy, device=None):
    device = device or next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        x = torch.tensor(np.asarray(X_noisy), dtype=torch.float32, device=device)
        return model(x).cpu().numpy()


def to_window_dict(arr):
    """Convert an (n_windows, window_size) array into the {idx: [values]}
    dict format used by qutsae.py's metric functions (mae_tswin, r2_tswin,
    ...), so classical and quantum results are scored identically."""
    return {i: list(arr[i]) for i in range(len(arr))}
