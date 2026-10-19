#!/bin/bash
# Start the traffic surveillance dashboard (creates the Python environment on first run).
set -e
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  # macOS's own /usr/bin/python3 is 3.9 - too old for current PyTorch. Use the newest Python 3.10+ available.
  PY=""
  for c in python3.13 python3.12 python3.11 python3.10 python3 python; do
    if command -v "$c" >/dev/null && "$c" -c 'import sys; sys.exit(sys.version_info < (3, 10))' 2>/dev/null; then
      PY="$c"; break
    fi
  done
  if [ -z "$PY" ]; then
    echo "Python 3.10 or newer is needed. Install it with:  brew install python@3.13  (or: uv python install 3.13)"
    exit 1
  fi
  echo "First run: creating .venv with $($PY --version) and installing packages (a few minutes)..."
  "$PY" -m venv .venv
  .venv/bin/pip install --upgrade pip -q
  .venv/bin/pip install -r requirements.txt
fi
exec .venv/bin/python app.py "$@"
