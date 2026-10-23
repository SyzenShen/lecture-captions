#!/bin/bash
# One-time setup: virtual environment, Python packages, offline en->zh model.
# (The Whisper model itself, ~460 MB, is downloaded automatically the first time you run.)
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"
"$PY" -m venv .venv
.venv/bin/pip install --quiet --upgrade pip
.venv/bin/pip install --quiet -r requirements.txt
.venv/bin/python fetch_argos_zh.py
chmod +x run.sh
echo
echo "Done. Start with:   ./run.sh --web --prompt-file prompts/example_biology.txt"
echo "macOS will ask for microphone access for your terminal the first time."
