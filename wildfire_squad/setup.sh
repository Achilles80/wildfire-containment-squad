#!/usr/bin/env sh
# One-time setup on macOS/Linux: create .venv and install everything (Python 3.12+ required).
set -e
cd "$(dirname "$0")"
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
# Download the demo's browser files now, so the demo also works offline later.
python prepare_demo_assets.py || true
echo "Setup complete. Next: ./run.sh demo | tests | experiments"
