#!/usr/bin/env bash
# Regenerate the manuscript figures (Figures 2-6) with the representative retrained networks
# (nn-results/representative_seed2/), as in the paper.
# Runs in a copy of the repository: the seed-2 checkpoints are copied over the released ones there,
# so the working tree is not modified.
# Usage:  reproducibility/figures.sh                       (dataset: DataSet/ in the repo root, or set DATASET)
#         CHECKPOINTS=released reproducibility/figures.sh  (use the released checkpoints instead)
# Output: runs/figures/v2_manuscript/figures/fig{2..6}_*_en.{png,pdf}, runs/figures/figs.log
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
CHECKPOINTS="${CHECKPOINTS:-representative_seed2}"
d="$RUNS_DIR/figures"
snapshot_repo "$d"
cd "$d"
if [ "$CHECKPOINTS" != "released" ]; then
    src="nn-results/$CHECKPOINTS"
    [ -d "$src" ] || { echo "Checkpoint folder $src not found" >&2; exit 1; }
    (cd "$src" && find . -type f ! -name README.md) | while read -r f; do cp "$src/$f" "nn-results/$f"; done
fi
echo "[$(date +%T)] generating figures with the $CHECKPOINTS checkpoints in $d"
MPLBACKEND=Agg "$PYTHON" generate_manuscript_figures.py --no-cache --dataset "$DATASET" > figs.log 2>&1
echo "[$(date +%T)] done: $d/v2_manuscript/figures/"
ls v2_manuscript/figures/*.png
