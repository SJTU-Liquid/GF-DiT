#!/usr/bin/env python3
"""Summarize serving-benchmark results produced by ae/gpu/07_policy_runs.sh.

Walks a results directory for runs (Plectra: sweep_<policy>_*, Legacy:
legacy_sp<N>_*) and prints one row per run with overall SLO attainment (failed
requests count as violations), mean / p95 latency of completed requests,
throughput, and per-class SLO.  A run is listed as FAILED if its server log exists
but it wrote no result JSON, or if the sweep / Legacy runner marked it failed
(<run>_failed.txt, e.g. the server died during the benchmark or the port was busy).  Exit status: 0 if all listed runs have results, 3 if some
are FAILED, 1 if there are no runs.  The directory may hold runs of earlier
invocations; 07_policy_runs.sh judges its own runs separately.  Standard library
only.

    python3 ae/gpu/summarize_runs.py ae/gpu/results/main-h20-wan
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def name_of(path: Path, suffix: str) -> str:
    return path.name.replace(suffix, "").replace("sweep_", "")


def row(path: Path) -> str:
    d = json.loads(path.read_text())
    pc = d.get("per_class", {})
    n = sum(c["n"] for c in pc.values())
    met = sum(c["slo_met"] for c in pc.values())
    slo = 100.0 * met / n if n else float("nan")
    cls = " ".join(f"{k}:{pc[k]['slo_met']}/{pc[k]['n']}" for k in sorted(pc))
    return (f"| {path.parent.name} | {name_of(path, '_result.json')} | {slo:.1f} | {d['latency_mean']:.1f} | "
            f"{d['latency_p95']:.1f} | {d['throughput_qps']:.4f} | "
            f"{d['completed_requests']}/{d['completed_requests'] + d['failed_requests']} | {cls} |")


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "ae/gpu/results")
    marked = {m.parent / m.name.replace("_failed.txt", ""): m.read_text().strip()
              for m in root.rglob("*_failed.txt")}  # written by the sweep / run_legacy on failure
    results = sorted(r for r in root.rglob("*_result.json")
                     if r.parent / r.name.replace("_result.json", "") not in marked)
    missing = sorted(log for log in root.rglob("*_server.log")
                     if not (log.parent / log.name.replace("_server.log", "_result.json")).is_file()
                     or log.parent / log.name.replace("_server.log", "") in marked)
    if not results and not missing:
        print(f"no runs under {root}")
        return 1
    print("| run | policy | SLO % | mean s | p95 s | QPS | completed | per-class SLO met |")
    print("|---|---|---:|---:|---:|---:|---:|---|")
    for f in results:
        print(row(f))
    for log in missing:
        why = marked.get(log.parent / log.name.replace("_server.log", ""), f"no result JSON; see {log}")
        print(f"| {log.parent.name} | {name_of(log, '_server.log')} | FAILED | | | | | {why} |")
    if missing:
        print(f"\n{len(missing)} run(s) FAILED")
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
