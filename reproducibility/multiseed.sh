#!/usr/bin/env bash
# Retrain the five networks (train_all.sh) for several random seeds and evaluate each
# seed on the test subjects. Each seed runs in its own copy of the repository under
# runs/seed_<s>/ so checkpoints never overwrite each other or the committed ones.
# Finished seeds (eval.log complete) are skipped, so the script can be resumed.
#
# Usage:  reproducibility/multiseed.sh [SEED ...]   (dataset: DataSet/ in the repo root, or set DATASET)
#         (default seeds: 42 1 2 3 4; about 40 min per seed on an RTX 5070)
# Then:   reproducibility/aggregate_seeds.py runs/seed_*
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
SEEDS=("$@"); [ ${#SEEDS[@]} -eq 0 ] && SEEDS=(42 1 2 3 4)
for s in "${SEEDS[@]}"; do
    d="$RUNS_DIR/seed_$s"
    if [ -f "$d/eval.log" ] && grep -q "RELATIVE IMPROVEMENTS" "$d/eval.log"; then
        echo "seed $s already done, skipping"; continue
    fi
    snapshot_repo "$d"
    cd "$d"
    echo "[$(date +%T)] seed $s: training (logs in $d/nn-results/train_logs/)"
    PYTHON="$PYTHON" DATASET="$DATASET" SEED="$s" ./train_all.sh > train_all.out 2>&1
    echo "[$(date +%T)] seed $s: evaluating"
    MPLBACKEND=Agg "$PYTHON" evaluate_full_neural_vs_analytical.py --dataset "$DATASET" > eval.log 2>&1
    echo "[$(date +%T)] seed $s: done"
    cd "$REPO"
done
echo "Aggregate with:  $PYTHON $REPO/reproducibility/aggregate_seeds.py $RUNS_DIR/seed_*"
