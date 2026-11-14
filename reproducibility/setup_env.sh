#!/usr/bin/env bash
# Create .venv with uv (https://docs.astral.sh/uv/) and install the dependencies.
# torch is installed with the CUDA build matching the local driver (CPU build if no GPU).
# Usage (from anywhere):  reproducibility/setup_env.sh
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
command -v uv >/dev/null || { echo "Install uv first: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2; exit 1; }
uv venv --python 3.12 .venv
uv pip install -p .venv --torch-backend=auto torch
uv pip install -p .venv -r requirements.txt
.venv/bin/python - <<'PY'
import torch, numpy, scipy, sys
print(f"python {sys.version.split()[0]}  torch {torch.__version__}  numpy {numpy.__version__}  scipy {scipy.__version__}")
print("CUDA:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "not available (CPU only)")
PY
