"""Precision and recall per rule: the system's cases against a hand-made list of the true violations.

    python training/score_accuracy.py ground_truth.csv violations.csv [--window 3]

ground_truth.csv  (fill it in while watching the video, BEFORE looking at the system's cases)
    time_s,code,class,note
    74.5,V2,Motorbike,red helmet rider weaving left lane
    102.0,V1,Bus,picks up passengers mid-road
  time_s = seconds into the video when the violation is clearest; code = V1..V8; class optional.

violations.csv = the dashboard's Export CSV (column video_time_s) or the command-line runner's violations.csv.

A case matches a true violation of the same rule within --window seconds (and the same class, when both give one);
each true violation matches at most one case. Unmatched cases = false alarms, unmatched true violations = misses.
"""

import argparse
import csv
from collections import defaultdict


def load(path, time_col):
    with open(path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        t = r.get(time_col) or r.get("video_time_s") or r.get("time_s")
        if t in (None, ""):
            continue
        out.append({"t": float(t), "code": r["code"].strip().upper(), "cls": (r.get("class") or r.get("cls") or "").strip(),
                    "id": r.get("id", ""), "note": r.get("note", "")})
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("truth")
    p.add_argument("cases")
    p.add_argument("--window", type=float, default=3.0, help="seconds between a case and the true violation")
    a = p.parse_args()
    truth, cases = load(a.truth, "time_s"), load(a.cases, "video_time_s")
    codes = sorted({x["code"] for x in truth} | {x["code"] for x in cases})
    print(f"{'rule':<5}{'true':>6}{'cases':>7}{'correct':>9}{'precision':>11}{'recall':>8}")
    misses, false_alarms = [], []
    for code in codes:
        tr = [x for x in truth if x["code"] == code]
        cs = sorted([x for x in cases if x["code"] == code], key=lambda x: x["t"])
        used = set()
        correct = 0
        for c in cs:
            best = None
            for i, g in enumerate(tr):
                if i in used or abs(g["t"] - c["t"]) > a.window:
                    continue
                if g["cls"] and c["cls"] and g["cls"].lower() != c["cls"].lower():
                    continue
                if best is None or abs(g["t"] - c["t"]) < abs(tr[best]["t"] - c["t"]):
                    best = i
            if best is None:
                false_alarms.append(c)
            else:
                used.add(best)
                correct += 1
        misses += [g for i, g in enumerate(tr) if i not in used]
        prec = f"{100 * correct / len(cs):.0f}%" if cs else "-"
        rec = f"{100 * correct / len(tr):.0f}%" if tr else "-"
        print(f"{code:<5}{len(tr):>6}{len(cs):>7}{correct:>9}{prec:>11}{rec:>8}")
    if misses:
        print("\nMissed (true violations with no case):")
        for g in misses:
            print(f"  {g['t']:7.1f} s  {g['code']}  {g['cls']}  {g['note']}")
    if false_alarms:
        print("\nFalse alarms (cases with no true violation):")
        for c in false_alarms:
            print(f"  {c['t']:7.1f} s  {c['code']}  {c['cls']}  case {c['id']}")


if __name__ == "__main__":
    main()
