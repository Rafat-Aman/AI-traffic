#!/usr/bin/env python3
"""Does the reasoner refuse when the photographs do not support the case?

A model that always agrees with the rule engine is worse than useless on an enforcement case: it launders a
false positive into a confident paragraph. This runs the insufficient-evidence probe described in the
project plan - cases whose evidence cannot support the finding - and reports whether the model declined.

    ./.venv/bin/python tools/vlm_probe.py
    ./.venv/bin/python tools/vlm_probe.py --backend cloud

Each probe says which verdicts are acceptable. "supported" on a probe whose evidence shows nothing of the
kind is a failure, and is the specific failure that matters.
"""

import argparse
import shutil
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from surveillance import vlm   # noqa: E402

try:
    import ultralytics
    ASSETS = Path(ultralytics.__file__).parent / "assets"
except ImportError:
    ASSETS = None

# (name, photograph, case, verdicts that count as passing, what is being tested)
PROBES = [
    ("unrelated-evidence", "zidane.jpg",
     {"code": "V1", "title": "Bus stopped away from a bus stop", "cls": "Bus", "class_conf": 0.91,
      "kph": 0.0, "zone": "Mirpur Road southbound", "wall_time": "2026-03-04 17:42:11",
      "details": {"dwell_s": 38}},
     {"insufficient", "contradicted"},
     "the photograph has no bus and no road in it at all"),
    ("wrong-class", "bus.jpg",
     {"code": "V6", "title": "Motorbike rider without helmet", "cls": "Motorbike", "class_conf": 0.88,
      "kph": 14.0, "zone": "Mirpur Road southbound", "wall_time": "2026-03-04 09:12:00", "details": {}},
     {"insufficient", "contradicted"},
     "the photograph shows a bus, not a bare-headed motorbike rider"),
    ("plausible-case", "bus.jpg",
     {"code": "V1", "title": "Bus stopped away from a bus stop", "cls": "Bus", "class_conf": 0.93,
      "kph": 0.0, "zone": "Mirpur Road southbound", "wall_time": "2026-03-04 17:42:11",
      "details": {"dwell_s": 42, "vehicles_passed": 19}},
     {"supported", "insufficient", "contradicted"},
     "a real stopped bus: any verdict is defensible, this one checks the answer reads sensibly"),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=["auto", "local", "cloud"], default="auto")
    ap.add_argument("--max-side", type=int, default=768)
    a = ap.parse_args()

    if ASSETS is None or not ASSETS.exists():
        print("The sample photographs ship with ultralytics, which is not importable here.")
        return 2
    settings = dict(vlm.DEFAULTS, vlm_backend=a.backend, vlm_max_side=a.max_side, vlm_max_images=1)
    backend = vlm.resolve_backend(settings)
    if backend is None:
        print("No backend reachable. Start ./tools/vlm_server.sh, or set ANTHROPIC_API_KEY.")
        return 2
    print(f"Probing {backend.name} / {backend.model}\n")

    tmp = Path(tempfile.mkdtemp())
    try:
        failures = 0
        explainer = vlm.Explainer(tmp, settings, lambda *x: None)
        for name, asset, case, accept, why in PROBES:
            shutil.copy(ASSETS / asset, tmp / asset)
            v = {"id": 0, "crop": asset, "snapshot": asset, "evidence": [], **case}
            t0 = time.perf_counter()
            ex = explainer.explain_now(v, settings)
            took = time.perf_counter() - t0
            if not ex.ok:
                print(f"  ERROR  {name}: {ex.error}")
                failures += 1
                continue
            ok = ex.assessment in accept
            failures += not ok
            print(f"  {'PASS' if ok else 'FAIL'}  {name}  ({took:.0f} s)")
            print(f"        probe        {why}")
            print(f"        acceptable   {', '.join(sorted(accept))}")
            print(f"        answered     {ex.assessment} at {ex.confidence:.0%} confidence")
            print(f"        summary      {ex.summary}")
            print(f"        cause        {ex.cause or '-'}\n")
        print(f"{len(PROBES) - failures}/{len(PROBES)} probes passed")
        if failures:
            print("\nA failure here means the model agreed with a finding its photographs do not support.\n"
                  "Treat its explanations as unreliable until this passes, and prefer the cloud backend.")
        return 1 if failures else 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
