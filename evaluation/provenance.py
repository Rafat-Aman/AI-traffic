"""Provenance stamped on every results/*.json: command, inputs, git commit, timestamp (read-only git)."""
import datetime
import hashlib
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def git_commit():
    try:
        c = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=10).stdout.strip()
        d = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, timeout=10).stdout
        return c + (" (+uncommitted changes)" if d.strip() else "")
    except Exception:
        return "unknown"


def file_stamp(path):
    p = Path(path)
    if not p.is_file():
        return {"path": str(path), "exists": p.exists()}
    size = p.stat().st_size
    h = hashlib.sha256()
    with open(p, "rb") as f:
        h.update(f.read(1 << 20))
    return {"path": str(path), "bytes": size, "sha256_first_1MiB": h.hexdigest()[:16]}


def make(argv=None, inputs=()):
    return {"command": " ".join(argv if argv is not None else sys.argv),
            "inputs": [file_stamp(i) for i in inputs],
            "git_commit": git_commit(),
            "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            "python": platform.python_version(), "machine": platform.processor() or platform.machine()}
