"""
Monolith QAE training/eval, adapted from the REAL QUANTICS 2026 source
(`qae_architectures/Jacob/ts_pl_v6_03_monolith_mcgen_tau30_jc_q6_l4_t2_x2_r4_0_pi4pi4_clean.ipynb`).

This is a straight extraction of that notebook's own functions (mackey_glass,
x2y, create_sw_tens, get_mini_batches, mse_cost_on_tensors, full_qae_shape,
full_qae, train_with_noise) into a headless, parameterized script - no
plotting, no debug cells, same math throughout. Only additions: a CLI, a
sigma/tau/seed sweep, and JSON result logging (the eval-metric computation
in "compute the eval metrics" below is also lifted straight from the
notebook's own final cells).

Why this exists: the real repo's eval CSVs (`Jacob/qae_eval_framework/*.csv`)
only cover sigma=0.2 (their one published noise level) - Reviewer 4's
camera-ready request for "1-2 additional noise levels beyond sigma=0.2" has
NO existing real data to reuse, so this script computes genuinely new
results using the paper's own real ansatz/training code, not a
reimplementation.

Usage:
    # smoke test: 1 seed, few epochs, fast
    python3 train_monolith_qae.py --tau 30 --sigma 0.2 --seed 2025 --n_epochs 20

    # one real run at their exact published config (~minutes, CPU)
    python3 train_monolith_qae.py --tau 30 --sigma 0.2 --seed 2025

    # the noise-ablation grid Reviewer 4 asked for (tau x sigma x seed)
    python3 sweep_monolith_qae.py

Requires: pennylane==0.40.0, pennylane-lightning==0.40.0 (see requirements.txt).
Runs on CPU (lightning.qubit) - no GPU needed, but each full run (300 epochs,
default config: window=6, n_latent=4, n_layers=4, Rxyz -> 8 wires) takes on
the order of minutes; use --n_epochs to shorten for a quick check.
"""
import argparse
import json
import os
import time

import pennylane as qml
import torch  # noqa: F401 (imported because the real notebook imports it - unused directly here)
from pennylane import numpy as np
from sklearn.metrics import mean_squared_error

from qae_utils.Window import ts_wind_split, ts_wind_flatten_avg  # noqa: F401 (flatten used below)
from qae_utils.Window import ts_add_noise, ts_wind_make


# ---------------------------------------------------------------------------
# Verbatim (headless) from the real notebook - x2y, mackey_glass, create_sw_tens,
# get_mini_batches, mse_cost_on_tensors, cost_fun_gen_on_tensors, full_qae_shape,
# full_qae, train_with_noise. Comments trimmed; logic unchanged.
# ---------------------------------------------------------------------------

def x2y(x, xlim=(0, 1), ylim=(0, np.pi)):
    low_x, high_x = xlim
    low_y, high_y = ylim
    input_range_length = high_x - low_x
    if np.isclose(input_range_length, 0.0):
        return (low_y + high_y) / 2
    scaling_factor = (high_y - low_y) / input_range_length
    return low_y + (x - low_x) * scaling_factor


def mackey_glass(length=1300, tau=17, beta=0.2, gamma=0.1, n=10, x0=1.2):
    """Real repo's own Mackey-Glass recursion (note: denominator uses
    xt_tau**n directly, i.e. theta=1 is implicit; NOT (xt_tau/theta)**n with
    a separately-stated theta - same thing here since theta=1)."""
    x = np.zeros(length + tau + 1, dtype=float)
    x[:tau + 1] = x0
    for t in range(tau, length + tau):
        xt = x[t]
        xt_tau = x[t - tau]
        dx = beta * xt_tau / (1.0 + xt_tau ** n) - gamma * xt
        x[t + 1] = xt + dx
    return x[tau + 1:]


def create_sw_tens(X, y, split, noise=0.0, wind_size=5, wind_step=2, range_low=0.2, range_high=0.8,
                    seed=0, noise_clip=True):
    y_noisy = ts_add_noise(y, noise=noise, noise_type='normal', clip=noise_clip,
                            range_low=range_low, range_high=range_high, seed=seed)
    y_ts = ts_wind_make(y_noisy, wind_size, wind_step)
    X_ts = ts_wind_make(X, wind_size, wind_step)
    X_train_ts, y_train_ts, X_test_ts, y_test_ts = ts_wind_split(X_ts, y_ts, split)

    X_train_tens = np.tensor(X_train_ts, requires_grad=False)
    y_train_tens = np.tensor(y_train_ts, requires_grad=False)
    X_test_tens = np.tensor(X_test_ts, requires_grad=False)
    y_test_tens = np.tensor(y_test_ts, requires_grad=False)
    return X_train_tens, y_train_tens, X_test_tens, y_test_tens


def get_mini_batches(W_noisy, W_clean, batch_size=10, shuffle=True, seed=0):
    if seed == 0:
        seed = int(time.time() * 1000) % 10000
    np.random.seed(seed)
    num_samples = W_clean.shape[0]
    indices = np.arange(num_samples)
    if shuffle:
        np.random.shuffle(indices)
    for i in range(0, num_samples, batch_size):
        batch_indices = indices[i:i + batch_size]
        yield W_noisy[batch_indices], W_clean[batch_indices]


def mse_cost_on_tensors(targets, predictions):
    cost = 0
    vals = 0
    for i in range(len(targets)):
        for w in range(len(targets[i])):
            cost = cost + (targets[i][w] - predictions[i][w]) ** 2
            vals += 1
    return cost / vals


def cost_fun_gen_on_tensors(model, cost_fun):
    def _cost_fun(params, inputs, targets):
        preds = [model(params, x) for x in inputs]
        return cost_fun(targets, preds)
    return _cost_fun


def full_qae_shape(n_latent, n_trash, n_extra=0, n_layers=1, rot='Ry'):
    n_wires = n_latent + n_trash + n_extra
    if rot == 'Ry':
        return qml.BasicEntanglerLayers.shape(n_layers=n_layers * 2, n_wires=n_wires)
    elif rot == 'Rxyz':
        return qml.StronglyEntanglingLayers.shape(n_layers=n_layers * 2, n_wires=n_wires)


def full_qae(wires, n_latent, n_trash, n_extra, n_layers=1, rot='Ry', add_outseq=True, invert_dec=True):
    n_data = n_latent + n_trash
    n_anz = n_data + n_extra
    n_zero = n_trash + n_extra
    latent_wires = wires[0:n_latent]
    trash_wires = wires[n_latent:n_latent + n_trash]
    extra_wires = wires[n_data:n_data + n_extra]
    zero_wires = wires[n_anz:n_anz + n_zero]
    data_wires = latent_wires + trash_wires
    anz_wires = latent_wires + trash_wires + extra_wires

    def _sequence_encoder(wires_, inputs):
        qml.AngleEmbedding(inputs, wires=wires_, rotation='Y')

    def _entangler_shape(n_layers_, n_wires_, rot_='Ry'):
        if rot_ == 'Ry':
            return qml.BasicEntanglerLayers.shape(n_layers=n_layers_, n_wires=n_wires_)
        elif rot_ == 'Rxyz':
            return qml.StronglyEntanglingLayers.shape(n_layers=n_layers_, n_wires=n_wires_)
        return ()

    def _entangler(wires_, weights, rot_='Ry'):
        if rot_ == 'Ry':
            qml.BasicEntanglerLayers(weights, wires=anz_wires, rotation=qml.RY)
        elif rot_ == 'Rxyz':
            qml.StronglyEntanglingLayers(weights, wires=anz_wires)

    def _swap(from_wires, to_wires):
        for i in range(len(from_wires)):
            qml.SWAP(wires=[from_wires[i], to_wires[i]])

    def _full_qae(weights, inputs):
        n_anz_wires = n_latent + n_trash + n_extra
        enc_weights_shape = _entangler_shape(n_layers, n_anz_wires, rot_=rot)
        dec_weights_shape = enc_weights_shape

        _sequence_encoder(data_wires, inputs)
        qml.Barrier(wires)

        enc_weights = weights[:n_layers].reshape(enc_weights_shape)
        dec_weights = weights[n_layers:].reshape(dec_weights_shape)

        _entangler(anz_wires, enc_weights, rot_=rot)
        qml.Barrier(wires)
        _swap(trash_wires + extra_wires, zero_wires)
        qml.Barrier(wires)

        if invert_dec:
            qml.adjoint(_entangler)(anz_wires, dec_weights, rot_=rot)
        else:
            _entangler(anz_wires, enc_weights, rot_=rot)
        qml.Barrier(wires)

        if add_outseq:
            qml.adjoint(_sequence_encoder)(data_wires, inputs)

        return [qml.expval(qml.PauliZ(wires=w)) for w in data_wires]
    return _full_qae


def train_with_noise(model, W, cost_fun, optimizer, n_epochs, init_weights, split,
                      log_interv=10, prompt_fract=0.1, seed=0,
                      wind_size=8, wind_step=4, noise=0, enc_lim=(0, 1), dec_lim=(0, 1),
                      noise_clip=True, batch_size=10, lr_decay_rate=0.75, lr_decay_steps=60):
    enc_low, enc_high = enc_lim
    dec_low, dec_high = dec_lim
    if seed == 0:
        seed = int(time.time() * 1000) % 10000
    np.random.seed(seed)

    hist_cost, hist_params = [], []
    params = init_weights.copy()

    _, y_pure, _, _ = create_sw_tens(W, W, split, noise=0,
                                      wind_size=wind_size, wind_step=wind_step,
                                      range_low=enc_low, range_high=enc_high)
    y_pure = x2y(y_pure, xlim=(enc_low, enc_high), ylim=(dec_low, dec_high))

    start_time = time.time()
    for it in range(n_epochs):
        _, X_noisy, _, _ = create_sw_tens(W, W, split, noise=noise, seed=seed + it, noise_clip=noise_clip,
                                           wind_size=wind_size, wind_step=wind_step,
                                           range_low=enc_low, range_high=enc_high)
        if (it + 1) % lr_decay_steps == 0:
            optimizer.stepsize *= lr_decay_rate

        batches = get_mini_batches(X_noisy, y_pure, batch_size=batch_size, shuffle=True, seed=seed + it)
        acc_batch_cost = 0
        batch_no = 0
        for batch_no, (noisy_batch, pure_batch) in enumerate(batches):
            params, batch_cost = optimizer.step_and_cost(lambda p: cost_fun(p, noisy_batch, pure_batch), params)
            acc_batch_cost += batch_cost
        cost = acc_batch_cost / (batch_no + 1)

        elapsed = time.time() - start_time
        if it % log_interv == 0:
            hist_cost.append(cost)
            hist_params.append(params)
        if (prompt_fract == 0) or (it % max(int(prompt_fract * n_epochs), 1) == 0):
            print(f"  epoch {it:03d} ({int(elapsed):04d}s) cost={cost:0.6f} "
                  f"min={float(np.min(hist_cost)):0.6f} lr={optimizer.stepsize:0.4f}")

    min_iter = int(np.argmin(hist_cost))
    min_cost = float(hist_cost[min_iter])
    opt_params = hist_params[min_iter]
    return opt_params, hist_cost, (min_iter, min_cost, time.time() - start_time)


# ---------------------------------------------------------------------------
# New: CLI driver + eval (delta_test), same metric definition as the real
# eval CSVs (Jacob/qae_eval_framework/eval_all_*.csv: imp_test_pct =
# 100*(1 - mse_recovered/mse_noisy), computed on overlap-averaged windows).
# ---------------------------------------------------------------------------

def run_one(tau, sigma, seed, n_epochs, wind_size=6, wind_step=2, n_latent=4, n_layers=4,
            rot="Rxyz", split=0.75, lr_initial=0.1, batch_size=10, weight_scaler=0.1,
            beta=0.17, gamma=0.1, n=10, x0=1.2, length=1200, noise_clip=True, log_interv=10,
            sim="lightning.qubit"):
    n_trash = wind_size - n_latent
    n_wires = n_latent + 2 * n_trash  # n_extra=0, matches the real notebook

    np.random.seed(seed)

    # data (real notebook's exact recipe: min-max normalise raw series to [0,1],
    # THEN downsample by 6 starting at offset 3 -> 200 points)
    data_y = mackey_glass(length=length, beta=beta, gamma=gamma, n=n, tau=tau, x0=x0)
    data_y = x2y(data_y, xlim=(np.min(data_y), np.max(data_y)), ylim=(0, 1))
    data_y = data_y[3::6]
    assert len(data_y) == 200, f"expected 200 points, got {len(data_y)}"

    y_enc_low, y_enc_high = 0.0, np.pi / 4
    y = x2y(data_y, xlim=(0, 1), ylim=(y_enc_low, y_enc_high))

    wires = list(range(n_wires))
    shape = full_qae_shape(n_latent, n_trash, n_layers=n_layers, rot=rot)
    n_weights = int(np.prod(shape))
    init_weights = np.random.uniform(high=2 * np.pi, size=shape, requires_grad=True) * weight_scaler

    qae = full_qae(wires, n_latent, n_trash, 0, n_layers=n_layers, rot=rot, add_outseq=False, invert_dec=True)
    dev = qml.device(sim, wires=n_wires, shots=None, seed=seed)
    qae_model = qml.QNode(qae, dev, interface="autograd", diff_method="adjoint")

    opt = qml.AdamOptimizer(stepsize=lr_initial, beta1=0.99)
    cost_fun = cost_fun_gen_on_tensors(qae_model, mse_cost_on_tensors)

    t0 = time.time()
    opt_params, hist_cost, (min_iter, min_cost, train_secs) = train_with_noise(
        qae_model, y, cost_fun, opt, n_epochs, init_weights, split,
        wind_size=wind_size, wind_step=wind_step, noise=sigma, log_interv=log_interv,
        prompt_fract=0.2, seed=seed, enc_lim=(y_enc_low, y_enc_high), dec_lim=(y_enc_low, y_enc_high),
        noise_clip=noise_clip, batch_size=batch_size,
    )
    elapsed = time.time() - t0

    # ---- eval on the TEST partition, same metric as the real eval CSVs ----
    X_train_pure, y_train_pure, X_test_pure, y_test_pure = create_sw_tens(
        y, y, split, noise=0, wind_size=wind_size, wind_step=wind_step,
        range_low=y_enc_low, range_high=y_enc_high)
    # FIXED 2026-08-30: eval noise now uses the "official" evaluation
    # framework's own fixed convention (Jacob/qae_eval_framework/
    # universal_framework_all.ipynb: seed=99123, clip=False) - which is what
    # actually produced the published eval CSVs / Table 2 numbers - instead
    # of this script's earlier `seed=seed+n_epochs, noise_clip=noise_clip`
    # (True), which varied per run and wasn't comparable across seeds/epoch
    # counts, nor to project_b_classical's eval_fixed_noise() (same
    # TEST_SEED_FIXED=99123, clip=False). Training-time noise (train_with_noise
    # call above) is UNCHANGED and still uses noise_clip=True, matching this
    # specific notebook's own training protocol - only the final eval
    # convention is standardised here, not the training convention.
    EVAL_SEED_FIXED = 99123
    _, y_train_noisy, _, y_test_noisy = create_sw_tens(
        y, y, split, noise=sigma, wind_size=wind_size, wind_step=wind_step,
        seed=EVAL_SEED_FIXED, noise_clip=False, range_low=y_enc_low, range_high=y_enc_high)

    test_qae = full_qae(wires, n_latent, n_trash, 0, n_layers=n_layers, rot=rot, add_outseq=False, invert_dec=True)
    test_model = qml.QNode(test_qae, dev, interface="autograd", diff_method="adjoint")
    pred_from_noisy_test = np.stack([test_model(opt_params, x) for x in y_test_noisy], requires_grad=False)

    y_test_pure_flat = x2y(ts_wind_flatten_avg(y_test_pure, wind_step), xlim=(y_enc_low, y_enc_high), ylim=(0, 1))
    y_test_noisy_flat = x2y(ts_wind_flatten_avg(y_test_noisy, wind_step), xlim=(y_enc_low, y_enc_high), ylim=(0, 1))
    pred_test_flat = x2y(ts_wind_flatten_avg(pred_from_noisy_test, wind_step), xlim=(y_enc_low, y_enc_high), ylim=(0, 1))

    mse_test_noise = mean_squared_error(y_test_pure_flat, y_test_noisy_flat)
    mse_test_rec = mean_squared_error(y_test_pure_flat, pred_test_flat)
    delta_test_pct = 100.0 * (1.0 - mse_test_rec / max(mse_test_noise, 1e-12))

    return {
        "tau": tau, "sigma": sigma, "seed": seed, "n_epochs": n_epochs,
        "wind_size": wind_size, "wind_step": wind_step, "n_latent": n_latent,
        "n_trash": n_trash, "n_layers": n_layers, "rot": rot,
        "train_noise_clip": noise_clip, "eval_seed_fixed": EVAL_SEED_FIXED, "eval_noise_clip": False,
        "n_weights": n_weights, "n_wires": n_wires,
        "min_train_cost": min_cost, "min_train_iter": min_iter,
        "mse_test_noise": float(mse_test_noise), "mse_test_recovered": float(mse_test_rec),
        "delta_test_pct": float(delta_test_pct), "elapsed_sec": round(elapsed, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", type=int, default=30, choices=[17, 30])
    ap.add_argument("--sigma", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--n_epochs", type=int, default=300)
    ap.add_argument("--wind_size", type=int, default=6)
    ap.add_argument("--n_latent", type=int, default=4)
    ap.add_argument("--n_layers", type=int, default=4)
    ap.add_argument("--rot", default="Rxyz", choices=["Ry", "Rxyz"])
    ap.add_argument("--out", default="results_qml/monolith_qml.json")
    args = ap.parse_args()

    row = run_one(args.tau, args.sigma, args.seed, args.n_epochs,
                   wind_size=args.wind_size, n_latent=args.n_latent,
                   n_layers=args.n_layers, rot=args.rot)

    print(f"\ntau={row['tau']} sigma={row['sigma']} seed={row['seed']} "
          f"n_weights={row['n_weights']} n_wires={row['n_wires']} "
          f"Delta_test={row['delta_test_pct']:+.2f}% ({row['elapsed_sec']:.1f}s)")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    rows = json.load(open(args.out)) if os.path.exists(args.out) else []
    rows.append(row)
    with open(args.out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"Appended to {args.out} ({len(rows)} row(s) total)")


if __name__ == "__main__":
    main()
