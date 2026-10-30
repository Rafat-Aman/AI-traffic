#!/usr/bin/env python3
"""Can this machine run the reasoning model, and what does it cost when it does?

Prints what the hardware offers, whether a backend is reachable, and - with --case - actually explains a
stored violation while watching the clock and the memory, so the answer is measured rather than guessed.

    ./.venv/bin/python tools/vlm_check.py              # hardware + backend only, instant
    ./.venv/bin/python tools/vlm_check.py --case 12    # explain case 12 and report what it cost
    ./.venv/bin/python tools/vlm_check.py --case 12 --backend cloud
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from surveillance import vlm, vlm_cascade         # noqa: E402
from surveillance.store import Store              # noqa: E402

DATA = ROOT / "data"
GB = 1024 ** 3


def meminfo():
    kv = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        k, _, v = line.partition(":")
        kv[k] = int(v.strip().split()[0]) * 1024
    return kv


def rss_of(pattern):
    """Resident memory of a running process whose command line matches, in bytes (0 if not running)."""
    for p in Path("/proc").iterdir():
        if not p.name.isdigit():
            continue
        try:
            if pattern not in (p / "cmdline").read_bytes().decode(errors="ignore").replace("\0", " "):
                continue
            for line in (p / "status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
        except OSError:
            continue
    return 0


class Peak(threading.Thread):
    """Watches a process's resident memory while the model is answering."""

    def __init__(self, pattern):
        super().__init__(daemon=True)
        self.pattern, self.peak, self.stop = pattern, 0, threading.Event()

    def run(self):
        while not self.stop.is_set():
            self.peak = max(self.peak, rss_of(self.pattern))
            self.stop.wait(0.25)


def hardware():
    m = meminfo()
    cpu = ""
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("model name"):
            cpu = line.split(":", 1)[1].strip()
            break
    flags = set(re.findall(r"\bavx512\w*|\bavx2\b", Path("/proc/cpuinfo").read_text()))
    physical = 0
    try:
        out = subprocess.run(["lscpu", "-p=Core,Socket"], capture_output=True, text=True).stdout
        physical = len({l for l in out.splitlines() if l and not l.startswith("#")})
    except OSError:
        pass
    free = shutil.disk_usage(ROOT).free
    nvidia = shutil.which("nvidia-smi") is not None and os.path.exists("/dev/nvidiactl")

    if any("vnni" in f for f in flags):
        simd = "AVX-512 + VNNI (the fast path for 4-bit weights)"
    elif any("512" in f for f in flags):
        simd = "AVX-512"
    elif "avx2" in flags:
        simd = "AVX2"
    else:
        simd = "baseline - expect slow inference"

    print("Hardware")
    print(f"  CPU            {cpu}")
    print(f"  cores          {physical or '?'} physical / {os.cpu_count()} logical")
    print(f"  SIMD           {simd}")
    print(f"  memory         {m['MemTotal'] / GB:.0f} GB total, {m['MemAvailable'] / GB:.0f} GB available, "
          f"{m.get('SwapTotal', 0) / GB:.0f} GB swap")
    print(f"  GPU            {'CUDA available' if nvidia else 'none usable (CPU inference)'}")
    print(f"  disk here      {free / GB:.0f} GB free on the project's partition")
    return m, physical


def model_files():
    d = ROOT / "models" / "vlm"
    w = sorted(d.glob("*.gguf"))
    weights = [p for p in w if not p.name.startswith("mmproj")]
    mmproj = [p for p in w if p.name.startswith("mmproj")]
    print("\nLocal model")
    if not weights or not mmproj:
        print("  not installed. Run ./tools/setup_vlm.sh")
        return 0
    total = sum(p.stat().st_size for p in weights + mmproj)
    for p in weights + mmproj:
        print(f"  {p.name}  {p.stat().st_size / GB:.1f} GB")
    return total


def verdict(mem, model_bytes, running_rss):
    """The question the team actually asks: will this fall over?"""
    avail = mem["MemAvailable"]
    # Measured on this project's setup: 3.1 GB of weights came to 4.7 GB resident once the KV cache for an
    # 8192-token context and the vision compute buffers were allocated, so the overhead is ~1.6 GB.
    need = running_rss or (model_bytes + 1.6 * GB)
    print("\nVerdict")
    if not model_bytes and not running_rss:
        print("  No local model installed, so nothing to size yet.")
        return
    print(f"  the reasoner needs about {need / GB:.1f} GB resident, against {avail / GB:.1f} GB available")
    if need > avail:
        print("  ⚠ TIGHT. Close other applications, or it will page to swap and answers will crawl.")
    elif need > avail * 0.6:
        print("  ✓ fits, with little room to spare. Avoid analysing video at the same time.")
    else:
        print(f"  ✓ comfortable: {(avail - need) / GB:.0f} GB would still be free while it answers.")
    print("  The reasoner only runs when you press Explain, so it never competes with the detector\n"
          "  unless you ask it to while a camera is analysing.")


def tiers(mem):
    """Per-tier memory check of the two-tier cascade: the 2B stays resident, the 8B is started on demand."""
    avail = mem["MemAvailable"]
    print("\nTiers (surveillance/vlm_cascade.py)")
    need = {}
    for tier, label, port in (("screen", "2B screen  (always on)", 8081), ("reason", "8B reason  (on demand)", 8082)):
        ok = vlm_cascade.tier_installed(tier)
        need[tier] = vlm_cascade.tier_need_bytes(tier) if ok else 0
        up = vlm_cascade.http_health(f"http://127.0.0.1:{port}")
        rss = rss_of(f"--port {port}") if up else 0
        print(f"  {label:<24} {'installed' if ok else 'NOT installed (tools/setup_vlm.sh cascade)':<10}"
              f" needs ~{need[tier] / GB:.1f} GB" + (f"  | serving on :{port}, {rss / GB:.1f} GB resident" if up else ""))
    if all(need.values()):
        # What is already resident counts as available only if it is one of these servers.
        both = need["screen"] + need["reason"]
        print(f"  available now: {avail / GB:.1f} GB.  8B alone: {'fits' if need['reason'] <= avail else 'DOES NOT FIT'}"
              f"; 2B + 8B together need ~{both / GB:.1f} GB: {'fits' if both <= avail else 'does not fit - the dashboard will refuse to start the 8B and say so'}")
    return need


def run_case(vid, backend, settings):
    store = Store(DATA / "surveillance.db")
    v = store.violation(vid)
    if not v:
        ids = [r["id"] for r in store.violations(limit=10)]
        print(f"\nNo case {vid}. Cases available: {ids or 'none yet - analyse a video first'}")
        return 1
    chosen = vlm.resolve_backend(settings)
    if chosen is None:
        print("\nNo backend reachable. Start ./tools/vlm_server.sh, or set ANTHROPIC_API_KEY.")
        return 1
    imgs = vlm.gather_images(v, DATA, settings["vlm_max_images"], settings["vlm_max_side"])
    print(f"\nExplaining case {vid}: {v['code']} {v['title']} ({v['cls']})")
    print(f"  backend        {chosen.name} / {chosen.model}")
    print(f"  photographs    {len(imgs)} at up to {settings['vlm_max_side']} px")
    watch = Peak("llama-server")
    watch.start()
    t0 = time.perf_counter()
    ex = vlm.Explainer(DATA, settings, lambda *a: None).explain_now(v, settings)
    took = time.perf_counter() - t0
    watch.stop.set()
    print(f"  answered in    {took:.1f} s" + (f"  (llama-server peak {watch.peak / GB:.1f} GB)"
                                              if watch.peak else ""))
    if not ex.ok:
        print(f"\n  FAILED: {ex.error}")
        return 1
    print(f"\n  assessment     {ex.assessment}  (confidence {ex.confidence:.0%})")
    for label, text in (("summary", ex.summary), ("before", ex.pre_event), ("event", ex.event),
                        ("after", ex.aftermath), ("cause", ex.cause)):
        if text:
            print(f"  {label:<14} {text}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", type=int, help="explain this violation id and measure what it costs")
    ap.add_argument("--backend", choices=["auto", "local", "cloud"], default="auto")
    ap.add_argument("--max-side", type=int, help="longest image side sent to the model")
    ap.add_argument("--images", type=int, help="how many evidence photographs to send")
    a = ap.parse_args()

    settings = dict(vlm.DEFAULTS, vlm_backend=a.backend)
    if a.max_side:
        settings["vlm_max_side"] = a.max_side
    if a.images:
        settings["vlm_max_images"] = a.images

    mem, _ = hardware()
    total = model_files()
    running = rss_of("llama-server")

    print("\nBackend")
    st = vlm.backend_status(settings)
    print(f"  local server   {'up at ' + st['local_url'] if st['local_up'] else 'not running'}"
          + (f"  ({running / GB:.1f} GB resident)" if running else ""))
    print(f"  cloud key      {'set' if st['cloud_key'] else 'not set (ANTHROPIC_API_KEY / OPENAI_API_KEY)'}")
    print(f"  would use      {st['backend'] or 'nothing - Explain will return 503'}")

    verdict(mem, total, running)
    tiers(mem)
    return run_case(a.case, a.backend, settings) if a.case else 0


if __name__ == "__main__":
    sys.exit(main())
