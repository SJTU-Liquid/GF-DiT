# SPDX-License-Identifier: Apache-2.0
"""Analyze per-request SLO lateness from a bench result.json.

Requires the result.json to carry a ``per_request`` list (added to
diffusion_benchmark_serving.py). For each request we have:
  latency_ms, slo_ms (client budget), lateness_ms = latency - slo (>0 => violated).

Reports, per class and overall:
  - met / violated / failed counts and SLO rate
  - lateness distribution over VIOLATED requests (p50/p90/p99/max), both in ms
    and as a fraction of the class budget (overshoot ratio)
  - a coarse severity histogram: how many violations are marginal (<=10% over),
    moderate (10-50%), or severe (>50% over) -- this is what distinguishes a
    knife-edge miss from structural starvation.

Usage: python analyze_lateness.py <result.json> [result2.json ...]
"""
from __future__ import annotations

import json
import sys
from statistics import median


def pct(vals: list[float], q: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    i = min(len(s) - 1, int(q * len(s)))
    return s[i]


def analyze(path: str) -> None:
    d = json.load(open(path))
    pr = d.get("per_request")
    if not pr:
        print(f"{path}: NO per_request data (re-run with the patched bench)")
        return

    print(f"\n=== {path} ===")
    print(f"duration={d.get('duration', 0):.1f}s  completed={d.get('completed_requests')}")

    by_class: dict[str, list[dict]] = {}
    for r in pr:
        by_class.setdefault(r.get("request_class") or "?", []).append(r)

    hdr = (
        f"{'cls':>3} {'n':>4} {'met':>4} {'viol':>4} {'fail':>4} {'SLO%':>6} "
        f"{'budget_s':>8} | viol lateness_s p50/p90/max  overshoot p50/p90/max | "
        f"marg/mod/sev"
    )
    print(hdr)
    print("-" * len(hdr))

    order = ["S", "M", "L"]
    classes = [c for c in order if c in by_class] + [
        c for c in sorted(by_class) if c not in order
    ]
    all_late_s: list[float] = []
    all_over: list[float] = []
    for c in classes:
        rows = by_class[c]
        n = len(rows)
        fail = sum(1 for r in rows if not r["success"])
        defined = [r for r in rows if r["slo_ms"] is not None and r["success"]]
        met = sum(1 for r in defined if r["slo_achieved"])
        viol = [r for r in defined if not r["slo_achieved"]]
        slo = 100.0 * met / len(defined) if defined else 0.0
        budget = median([r["slo_ms"] for r in defined]) / 1000.0 if defined else 0.0

        late_s = [r["lateness_ms"] / 1000.0 for r in viol if r["lateness_ms"] is not None]
        # overshoot ratio = lateness / budget
        over = [
            (r["lateness_ms"] / r["slo_ms"])
            for r in viol
            if r["lateness_ms"] is not None and r["slo_ms"]
        ]
        all_late_s += late_s
        all_over += over
        marg = sum(1 for o in over if o <= 0.10)
        mod = sum(1 for o in over if 0.10 < o <= 0.50)
        sev = sum(1 for o in over if o > 0.50)

        if late_s:
            lat_str = f"{pct(late_s, .5):>5.1f}/{pct(late_s, .9):>5.1f}/{max(late_s):>5.1f}"
            ovr_str = f"{pct(over, .5):>5.0%}/{pct(over, .9):>5.0%}/{max(over):>5.0%}"
        else:
            lat_str = "   -  /  -  /  -"
            ovr_str = "  -  /  -  /  -"
        print(
            f"{c:>3} {n:>4} {met:>4} {len(viol):>4} {fail:>4} {slo:>5.1f}% "
            f"{budget:>8.1f} | {lat_str}  {ovr_str} | {marg:>2}/{mod:>2}/{sev:>2}"
        )

    nv = len(all_over)
    if nv:
        marg = sum(1 for o in all_over if o <= 0.10)
        mod = sum(1 for o in all_over if 0.10 < o <= 0.50)
        sev = sum(1 for o in all_over if o > 0.50)
        print("-" * len(hdr))
        print(
            f"ALL violations n={nv}: lateness_s p50={pct(all_late_s, .5):.1f} "
            f"p90={pct(all_late_s, .9):.1f} max={max(all_late_s):.1f} | "
            f"overshoot p50={pct(all_over, .5):.0%} p90={pct(all_over, .9):.0%} | "
            f"marginal(<=10%)={marg} moderate(10-50%)={mod} severe(>50%)={sev}"
        )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    for p in sys.argv[1:]:
        analyze(p)
