#!/usr/bin/env bash
# Classical track: tuned TinyMLP / MLP-AE / LSTM-AE, 4 series x 10 noise levels x 7 seeds
# (seeds 2023..2029 are FINAL_SEEDS inside run_classical_tuned_b.py).
#   Stage 1 (hyperparameter search) always runs at sigma=0.2 and is skipped
#   once its rows exist; Stage 2 (final) runs at the sigma given by --sigma.
# Resumable. Run it from any directory; do not start two copies at once.
#
#   WORKERS=24 PY=~/venv/bin/python3 ./run_all_classical.sh
set -euo pipefail
cd "$(dirname "$0")"

PY="${PY:-python3.11}"
WORKERS="${WORKERS:-4}"
DATASETS="${DATASETS:-mackey_glass_tau17,mackey_glass_tau30,energy,finance}"
SIGMAS="${SIGMAS:-0.0 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.45 0.50}"

for SIGMA in $SIGMAS; do
    echo "=== classical: sigma=${SIGMA} ==="
    "$PY" -u run_classical_tuned_b.py --datasets "$DATASETS" \
        --sigma "$SIGMA" --workers "$WORKERS"
done
echo "Classical grid done."
