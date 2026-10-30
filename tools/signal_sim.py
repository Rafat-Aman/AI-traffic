#!/usr/bin/env python3
"""Does measured-state signal control beat fixed-time, in simulation? And what does an ambulance cost the others?

    ./.venv/bin/python tools/signal_sim.py                    # tables to stdout
    ./.venv/bin/python tools/signal_sim.py --quick            # 2 seeds, one load level
    ./.venv/bin/python tools/signal_sim.py --out results.json # the same numbers as JSON (stdout stays tables)

The scenario is a 3x3 grid of signalised junctions with asymmetric, time-varying demand: an evening peak on the
middle street (the Mirpur Road role), heavy eastbound and moderate westbound, light parallel streets, and cross
streets heavier southbound than northbound. Three controllers face identical arrivals (same seeds):

  fixed (equal)   equal splits of an 80 s cycle - what a plan nobody has retuned looks like
  fixed (tuned)   splits proportional to each approach's AVERAGE flow over the run (cycle left at 80 s, not
                  optimised): a plan that knows the long-run average, so the fairer baseline, but cannot follow
                  the peak. A cycle length chosen by Webster's method might do somewhat better
  max pressure    Varaiya 2013, minimum green 15 s, maximum green 60 s, 4 s clearance

Then an emergency-vehicle run on the better controller: none vs hard corridor (EMVLight-style pre-emption) vs soft
priority (Cao-style weighted term), five ambulances crossing the corridor during the peak, measured on THEIR travel
time and on everyone else's mean delay - the question being whether pre-emption needs to hurt ordinary traffic.

What this is not: it is a mesoscopic simulator written for this project (see surveillance/optimize/sim.py), not
SUMO and not a Dhaka measurement. Demand levels were chosen to put the corridor near capacity; the max-pressure
minimum green was swept on seeds 0-2 of this scenario and the tables here use held-out seeds (10 and up). Nothing
here has controlled a real signal.
"""
import argparse
import json
import statistics as st
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from surveillance.optimize import network as N  # noqa: E402
from surveillance.optimize import routing, signals as S, sim as M  # noqa: E402

HORIZON = 1800.0
MAIN = 0.8                       # load level used for the emergency-vehicle runs


def build_network():
    return N.grid(3, 3, lat0=23.8040, lon0=90.3640, name="3x3 corridor demo (synthetic layout)")


def peak_flows(scale: float = 1.0) -> list:
    """Evening peak on the middle street; see the module docstring. Rates are vehicles per hour per flow."""
    east = [(0, 350), (300, 600), (600, 900), (1200, 600), (1500, 350)]
    west = [(0, 250), (600, 450), (1200, 250)]
    f = [M.Flow("W1", "E1", [(t, v * scale) for t, v in east]), M.Flow("E1", "W1", [(t, v * scale) for t, v in west])]
    for r in (0, 2):
        f += [M.Flow.constant(f"W{r}", f"E{r}", 150 * scale), M.Flow.constant(f"E{r}", f"W{r}", 150 * scale)]
    for c in range(3):
        f += [M.Flow.constant(f"N{c}", f"S{c}", 220 * scale), M.Flow.constant(f"S{c}", f"N{c}", 120 * scale)]
    return f


def controllers(net, flows):
    avg = M.expected_link_flows(net, flows, HORIZON)
    return {"fixed (equal)": lambda: S.FixedTimeController(net),
            "fixed (tuned)": lambda: S.FixedTimeController.from_flows(net, avg),
            "max pressure": lambda: S.MaxPressureController(net, min_green=15, max_green=60)}


def ms(xs):
    return f"{st.mean(xs):.1f} ± {st.pstdev(xs):.1f}" if len(xs) > 1 else f"{xs[0]:.1f}"


def compare(scales, seeds):
    net, out = build_network(), {}
    for sc in scales:
        flows = peak_flows(sc)
        out[sc] = {}
        for name, mk in controllers(net, flows).items():
            rs = [M.simulate(net, flows, mk(), horizon_s=HORIZON, seed=s) for s in seeds]
            out[sc][name] = [{k: r[k] for k in ("generated", "finished", "unfinished", "mean_delay_s", "p95_delay_s",
                                                "throughput_vph", "max_queue_veh", "forced_releases")} for r in rs]
    return out


def print_comparison(res, seeds):
    for sc, by in res.items():
        gen = st.mean(r["generated"] for r in next(iter(by.values())))
        print(f"\n### Load x{sc} (about {gen:.0f} vehicles per run, {len(seeds)} seeds, mean ± sd over seeds)\n")
        print("| controller | mean delay (s) | 95th pct delay (s) | throughput (veh/h) | max queue (veh) | unfinished |")
        print("|---|---|---|---|---|---|")
        for name, rs in by.items():
            print(f"| {name} | {ms([r['mean_delay_s'] for r in rs])} | {ms([r['p95_delay_s'] for r in rs])} | "
                  f"{ms([r['throughput_vph'] for r in rs])} | {ms([r['max_queue_veh'] for r in rs])} | "
                  f"{st.mean(r['unfinished'] for r in rs):.1f} |")
        mp = [r["mean_delay_s"] for r in by["max pressure"]]
        for base in ("fixed (equal)", "fixed (tuned)"):
            b = [r["mean_delay_s"] for r in by[base]]
            wins = sum(m < x for m, x in zip(mp, b))
            print(f"\nmax pressure vs {base}: mean delay {100 * (st.mean(mp) / st.mean(b) - 1):+.0f}%, "
                  f"lower in {wins} of {len(seeds)} seeds")


def ev_runs(seeds):
    net = build_network()
    flows = peak_flows(MAIN)
    ff = {l: x.free_flow_s for l, x in net.links.items()}
    east = routing.shortest_path(net, "W1", "E1", weights=ff).links
    west = routing.shortest_path(net, "E1", "W1", weights=ff).links
    evs = lambda: [M.EV(t, o, d, r) for t, (o, d, r) in zip(
        (600, 800, 1000, 1200, 1400), [("W1", "E1", east), ("E1", "W1", west)] * 3)]
    avg = M.expected_link_flows(net, flows, HORIZON)
    bases = {"max pressure": lambda: S.MaxPressureController(net, min_green=15, max_green=60),
             "fixed (tuned)": lambda: S.FixedTimeController.from_flows(net, avg)}
    out = {}
    for bname, mkbase in bases.items():
        variants = {"no priority": mkbase,
                    "hard corridor": lambda mk=mkbase: S.EmergencyCorridor(mk(), net)}
        if bname == "max pressure":
            variants["soft priority"] = lambda: S.SoftPriority(net, min_green=15, max_green=60)
        out[bname] = {}
        for vname, mk in variants.items():
            rs = [M.simulate(net, flows, mk(), horizon_s=HORIZON, seed=s, evs=evs()) for s in seeds]
            out[bname][vname] = [{"ev_travel_s": st.mean(e["travel_time_s"] for e in r["ev"]),
                                  "ev_unfinished": sum(not e["finished"] for e in r["ev"]),
                                  "others_mean_delay_s": r["mean_delay_s"], "others_p95_delay_s": r["p95_delay_s"],
                                  "unfinished": r["unfinished"]} for r in rs]
    return out


def print_ev(res, seeds):
    print(f"\n### Five ambulances along the middle street during the peak (load x{MAIN}, {len(seeds)} seeds)\n")
    print("| base controller | priority | EV travel time (s) | EV vs none | others' mean delay (s) | others vs none |")
    print("|---|---|---|---|---|---|")
    for bname, by in res.items():
        none = by["no priority"]
        for vname, rs in by.items():
            e, o = [r["ev_travel_s"] for r in rs], [r["others_mean_delay_s"] for r in rs]
            e0, o0 = [r["ev_travel_s"] for r in none], [r["others_mean_delay_s"] for r in none]
            d = lambda a, b: "-" if vname == "no priority" else f"{100 * (st.mean(a) / st.mean(b) - 1):+.1f}%"
            print(f"| {bname} | {vname} | {ms(e)} | {d(e, e0)} | {ms(o)} | {d(o, o0)} |")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--seed0", type=int, default=10, help="first seed (0-2 were used to tune the minimum green)")
    ap.add_argument("--scales", default="0.6,0.8,1.0", help="load levels, comma separated")
    ap.add_argument("--quick", action="store_true", help="2 seeds, load 0.8 only")
    ap.add_argument("--no-ev", action="store_true")
    ap.add_argument("--out", help="write the numbers as JSON here")
    a = ap.parse_args()
    seeds = list(range(a.seed0, a.seed0 + (2 if a.quick else a.seeds)))
    scales = [0.8] if a.quick else [float(x) for x in a.scales.split(",")]
    t0 = time.time()
    print("# Signal control in simulation (synthetic 3x3 grid; nothing here has driven a real signal)")
    res = compare(scales, seeds)
    print_comparison(res, seeds)
    ev = None
    if not a.no_ev:
        ev = ev_runs(seeds)
        print_ev(ev, seeds)
    print(f"\n(ran in {time.time() - t0:.0f} s)")
    if a.out:
        Path(a.out).write_text(json.dumps({"seeds": seeds, "scales": scales, "horizon_s": HORIZON,
                                           "comparison": {str(k): v for k, v in res.items()}, "ev": ev}, indent=1))


if __name__ == "__main__":
    main()
