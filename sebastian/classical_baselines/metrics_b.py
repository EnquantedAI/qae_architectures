"""Reconstruction merging + metrics for Track B, matching QUANTICS 2026's
own description (Sec. 4): "As the windows overlap (stride < W), the
resulting QAE reconstructions also overlap. The final time series
reconstruction is achieved by AVERAGING the overlapping values" - unlike
Track A's simple concatenation (merged_tswind in project/qutsae.py), which
assumed non-overlapping (or discarded) windows.
"""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def merge_overlapping_windows(windows, stride):
    """windows: (n_windows, window_size) array, in chronological order,
    starting at index 0 with the given stride. Returns the merged 1D series
    (length = stride*(n_windows-1) + window_size), each point the mean of
    every window that covers it."""
    n_windows, window_size = windows.shape
    total_len = stride * (n_windows - 1) + window_size
    sums = np.zeros(total_len)
    counts = np.zeros(total_len)
    for i in range(n_windows):
        s = i * stride
        sums[s:s + window_size] += windows[i]
        counts[s:s + window_size] += 1
    return sums / counts


def window_metrics(true_windows, pred_windows, stride):
    """Merge both true and predicted windows (overlap-averaged) and compute
    R2/RMSE/MAE on the merged series - the same quantity QUANTICS 2026's own
    Delta_test noise-removal percentage is derived from."""
    true_merged = merge_overlapping_windows(np.asarray(true_windows), stride)
    pred_merged = merge_overlapping_windows(np.asarray(pred_windows), stride)
    return {
        "r2": float(r2_score(true_merged, pred_merged)),
        "rmse": float(np.sqrt(mean_squared_error(true_merged, pred_merged))),
        "mae": float(mean_absolute_error(true_merged, pred_merged)),
    }


def delta_improvement(noisy_windows, pred_windows, true_windows, stride):
    """QUANTICS-2026-style relative noise-removal percentage, matching the
    real repo's own formula EXACTLY (verified against
    Jacob/qae_eval_framework/eval_all_*.csv's `imp_test_pct` column and the
    real training notebooks' own print statements):

        Delta = 100 * (1 - MSE(pred, true) / MSE(noisy, true))

    NOTE (fixed 2026-08-30): this was previously computed on MAE, not MSE -
    a real bug found while auditing this code before trusting it for the
    paper. MAE and
    MSE give different percentages for the same run, so every
    `delta_test_pct` value computed before this fix is NOT directly
    comparable to the real repo's own Delta_test numbers and needs
    recomputing (cheap - this only needs the eval pass rerun, but since model
    weights weren't saved, in practice means rerunning run_classical_grid_b.py
    - fast for classical models, unlike the QAE runs)."""
    true_merged = merge_overlapping_windows(np.asarray(true_windows), stride)
    noisy_merged = merge_overlapping_windows(np.asarray(noisy_windows), stride)
    pred_merged = merge_overlapping_windows(np.asarray(pred_windows), stride)
    mse_noisy = mean_squared_error(true_merged, noisy_merged)
    mse_pred = mean_squared_error(true_merged, pred_merged)
    if mse_noisy == 0:
        return 0.0
    return float(100.0 * (1.0 - mse_pred / mse_noisy)), float(mse_noisy), float(mse_pred)
