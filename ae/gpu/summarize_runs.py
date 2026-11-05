#!/usr/bin/env python3
"""Summarize serving-benchmark results produced by ae/gpu/07_policy_runs.sh.

Walks a results directory for *_result.json files (Plectra: sweep_<policy>_result.json,
Legacy: legacy_sp<N>_result.json) and prints one row per run with overall SLO
attainment (failed requests count as violations), mean / p95 latency of completed
requests, throughput, and per-class SLO.  Standard library only.

    python3 ae/gpu/summarize_runs.py ae/gpu/results/main
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def row(path: Path) -> str:
    d = json.loads(path.read_text())
    pc = d.get("per_class", {})
    n = sum(c["n"] for c in pc.values())
    met = sum(c["slo_met"] for c in pc.values())
    slo = 100.0 * met / n if n else float("nan")
    cls = " ".join(f"{k}:{pc[k]['slo_met']}/{pc[k]['n']}" for k in sorted(pc))
    name = path.name.replace("_result.json", "").replace("sweep_", "")
    return (f"| {path.parent.name} | {name} | {slo:.1f} | {d['latency_mean']:.1f} | {d['latency_p95']:.1f} | "
            f"{d['throughput_qps']:.4f} | {d['completed_requests']}/{d['completed_requests'] + d['failed_requests']} | {cls} |")


def main() -> None:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "ae/gpu/results")
    files = sorted(root.rglob("*_result.json"))
    if not files:
        sys.exit(f"no *_result.json under {root}")
    print("| run | policy | SLO % | mean s | p95 s | QPS | completed | per-class SLO met |")
    print("|---|---|---:|---:|---:|---:|---:|---|")
    for f in files:
        print(row(f))


if __name__ == "__main__":
    main()
