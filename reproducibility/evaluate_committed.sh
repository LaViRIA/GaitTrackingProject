#!/usr/bin/env bash
# Evaluate the committed checkpoints on the test subjects (Table of results in the paper).
# Runs in a copy of the repository so the committed result files are not overwritten.
# Usage:  reproducibility/evaluate_committed.sh   (dataset: DataSet/ in the repo root, or set DATASET)
# Output: runs/committed/eval.log   (compare with nn-results/full_neural_vs_analytical/results-v4.txt)
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
d="$RUNS_DIR/committed"
snapshot_repo "$d"
cd "$d"
echo "[$(date +%T)] evaluating committed checkpoints in $d"
MPLBACKEND=Agg "$PYTHON" evaluate_full_neural_vs_analytical.py --dataset "$DATASET" > eval.log 2>&1
echo "[$(date +%T)] done: $d/eval.log"
grep -A12 "MPJPE (95th percentile), global frame" eval.log || true
