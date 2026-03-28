#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# ── check for uv ──────────────────────────────────────────────────────────────
if ! command -v uv &>/dev/null; then
    echo "[setup] uv not found — install it first:"
    echo "        curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# ── create venv if needed ─────────────────────────────────────────────────────
if [ ! -d .venv ]; then
    echo "[setup] creating virtual environment …"
    uv venv
fi

# ── install dependencies ──────────────────────────────────────────────────────
echo "[setup] installing dependencies …"
uv pip install -r requirements.txt

echo ""
echo "[setup] done. Run the app with:  ./run.sh"
