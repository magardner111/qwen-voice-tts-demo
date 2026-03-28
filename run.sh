#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# ── sanity checks ─────────────────────────────────────────────────────────────
if [ ! -d .venv ]; then
    echo "[run] virtual environment not found — run ./setup.sh first"
    exit 1
fi

# ── detect best torch device ──────────────────────────────────────────────────
if [[ "$(uname -s)" == "Darwin" ]]; then
    DEVICE="mps"
elif .venv/bin/python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    DEVICE="cuda:0"
else
    DEVICE="cpu"
fi

echo "[run] device: $DEVICE"

# ── launch (Ctrl-C or closing the app window exits cleanly) ───────────────────
exec .venv/bin/python -u gui.py --device "$DEVICE" "$@"
