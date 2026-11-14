# Shared helpers for the reproducibility scripts (sourced, not executed).
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$REPO/.venv/bin/python}"
RUNS_DIR="${RUNS_DIR:-$REPO/runs}"

DATASET="${DATASET:-$REPO/DataSet}"
if [ ! -d "$DATASET" ]; then
    echo "Dataset not found at $DATASET." >&2
    echo "Unzip DataSet_GaitFromIMU.zip and move DataSet_Final/DataSet/ to the repository root as DataSet/," >&2
    echo "or set DATASET to the folder that holds one sub-folder per subject, e.g.  DATASET=/data/DataSet $0 ..." >&2
    exit 1
fi
DATASET="$(cd "$DATASET" && pwd)"
[ -x "$PYTHON" ] || { echo "Python not found at $PYTHON (run reproducibility/setup_env.sh or set PYTHON)" >&2; exit 1; }

# Copy the repository (code, data_split.json, committed checkpoints) into $1 so that
# training/evaluation, which write to ./nn-results, never touch the working tree.
snapshot_repo() {
    local dest="$1"
    rm -rf "$dest"; mkdir -p "$dest"
    tar -C "$REPO" -cf - \
        --exclude=./.git --exclude=./.venv --exclude=./runs --exclude=./DataSet --exclude='__pycache__' \
        --exclude=./v2_manuscript --exclude=./Results . | tar -C "$dest" -xf -
}
