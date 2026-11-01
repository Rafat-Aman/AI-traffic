#!/usr/bin/env bash
# Copy this working tree to the test server and restart the dashboard there. Run ON YOUR LAPTOP:
#
#   ./deploy/push.sh azureuser@20.198.10.5
#
# Copies the folder as it is on disk, uncommitted changes included, so there is no need to commit or push to GitHub
# first. Also copies the gitignored weights the server needs (ONNX detector exports, ReID .pth). Never copies:
# .venv, raw footage (videos/, *.mp4), the local data/ folder, the multi-GB VLM weights or vendor/.
#
# The first time, follow with:  ssh <host> bash dhaka-traffic/deploy/setup.sh <host name>
# After that, this script alone updates the code (and Python packages, if requirements.txt changed) and restarts.
set -euo pipefail

HOST="${1:-}"
if [[ -z "$HOST" ]]; then
    echo "usage: $0 user@server" >&2
    exit 2
fi
cd "$(dirname "$0")/.."

# Leading "/" anchors a pattern to the project root (tests/data/ is needed, data/ is not). Excluded paths are also
# protected from --delete, so the server's .venv and the ONNX files it exports stay put.
# *.mlpackage: CoreML only predicts on macOS, and find_helmet_model() picks helmet_cls.mlpackage over helmet_cls.pt
# whenever it exists, so on Linux V6 would fail on every crop. Without it the server uses the .pt.
# --chmod: this checkout sits on an NTFS mount where every file reads as 777; do not carry that over.
rsync -az --delete --info=progress2 --chmod=D755,F644 \
    --exclude=/.git/ --exclude=/.venv/ --exclude=__pycache__/ --exclude=.pytest_cache/ \
    --exclude=/data/ --exclude=/videos/ --exclude=/runs/ --exclude=/vendor/ --exclude=/API \
    --exclude='/models/vlm/*.gguf' --exclude='/models/vlm/*.sha256ok' \
    --exclude='*.mlpackage/' \
    --exclude='*.mp4' --exclude='*.avi' --exclude='*.mov' --exclude='*.mkv' --exclude='*.webm' \
    --exclude=.env --exclude=.DS_Store \
    ./ "$HOST:dhaka-traffic/"

ssh "$HOST" 'set -e
cd ~/dhaka-traffic
chmod +x start.sh tools/*.sh deploy/*.sh 2>/dev/null || true
if [ -f /etc/systemd/system/dhaka-traffic.service ] && [ -x .venv/bin/pip ]; then
    .venv/bin/pip install -q -r requirements.txt
    sudo systemctl restart dhaka-traffic
    echo "Code updated, dashboard restarting (about a minute until it answers)."
else
    echo "Code copied. First time on this server? Run:"
    echo "  ssh '"$HOST"' bash dhaka-traffic/deploy/setup.sh <public host name>"
fi'
