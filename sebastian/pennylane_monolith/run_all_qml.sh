#!/usr/bin/env bash
# Quantum track: full Monolith QAE grid, 3 series x 10 noise levels x 7 seeds.
#   series : Mackey-Glass (tau 17 and 30), energy, finance
#   sigma  : 0, 0.10, 0.15, ..., 0.50
#   seeds  : 2023..2029
# Resumable: rows already in results_qml/monolith_qml.json are skipped, so
# rerunning only computes what is missing. Run it from any directory.
# Do NOT start two copies at once - each call rewrites the whole JSON file.
#
#   WORKERS=28 PY=~/venv/bin/python3 ./run_all_qml.sh
set -euo pipefail
cd "$(dirname "$0")"

PY="${PY:-python3}"
WORKERS="${WORKERS:-28}"
SEEDS="${SEEDS:-2023,2024,2025,2026,2027,2028,2029}"
SIGMAS="${SIGMAS:-0,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50}"

for DATASET in mackey_glass energy finance; do
    echo "=== QAE: dataset=${DATASET} ==="
    "$PY" -u sweep_monolith_qae.py --dataset "$DATASET" \
        --sigmas "$SIGMAS" --seeds "$SEEDS" --workers "$WORKERS"
done
echo "QAE grid done."
