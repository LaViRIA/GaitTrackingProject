#!/usr/bin/env bash
# Execution-time benchmark (benchmark_timing.py) with the committed checkpoints.
# Run it twice on an otherwise idle machine; the two JSON files should agree within a few %.
# Usage:  reproducibility/timing.sh [extra benchmark_timing.py args]   (dataset: DataSet/ in the repo root, or set DATASET)
# Output: runs/timing/timing_<host>_run{1,2}.{json,log}
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
out="$RUNS_DIR/timing"; mkdir -p "$out"
cd "$REPO"
for r in 1 2; do
    f="$out/timing_$(hostname -s)_run$r"
    echo "[$(date +%T)] timing run $r -> $f.json"
    "$PYTHON" benchmark_timing.py --dataset "$DATASET" --out "$f.json" "$@" 2>&1 | tee "$f.log"
done
