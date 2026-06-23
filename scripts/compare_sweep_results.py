#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Summarize per-policy sweep results into a side-by-side table.

Reads each /tmp/sweep_${policy}_result.json + /tmp/sweep_${policy}_server.log
written by scripts/sweep_policies.sh. Prints per-class SLO and migration
counts so we can validate scheduler-policy changes (e.g. the sticky-rank
helper that closes the best_fit vs greedy white-migration gap).

Usage:
  scripts/compare_sweep_results.py edf_greedy edf_best_fit
  scripts/compare_sweep_results.py --out-dir /tmp --policies edf_greedy fcfs
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def _count_migrations(server_log: Path) -> int:
    if not server_log.exists():
        return -1
    text = server_log.read_text(errors="ignore")
    # data_plane.py logs one "runtime_v2 migrate done:" per completed migration.
    return len(re.findall(r"runtime_v2 migrate done:", text))


def _summarize(result_path: Path, server_log: Path) -> dict:
    out: dict = {"result_path": str(result_path), "ok": False}
    if not result_path.exists():
        return out
    try:
        data = json.loads(result_path.read_text())
    except Exception as exc:
        out["error"] = repr(exc)
        return out
    out["ok"] = True
    out["duration_s"] = float(data.get("duration", 0.0))
    out["completed"] = int(data.get("completed_requests", 0))
    out["per_class"] = {}
    for cls, stats in (data.get("per_class") or {}).items():
        out["per_class"][cls] = {
            "n": int(stats.get("n", 0)),
            "slo_met": int(stats.get("slo_met", 0)),
            "slo_rate": float(stats.get("slo_attainment_rate", 0.0)),
            "latency_mean": float(stats.get("latency_mean", 0.0)),
            "latency_p95": float(stats.get("latency_p95", 0.0)),
        }
    out["migrations"] = _count_migrations(server_log)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("policies", nargs="*", help="policy names to summarize")
    parser.add_argument("--out-dir", default="/tmp")
    parser.add_argument(
        "--policies",
        dest="policies_flag",
        nargs="*",
        default=None,
        help="alt form of positional policies",
    )
    args = parser.parse_args()
    policies = args.policies_flag if args.policies_flag is not None else args.policies
    if not policies:
        policies = ["edf_best_fit", "edf_greedy", "fcfs", "srtf"]

    rows: list[tuple[str, dict]] = []
    for p in policies:
        rows.append((p, _summarize(Path(args.out_dir) / f"sweep_{p}_result.json",
                                   Path(args.out_dir) / f"sweep_{p}_server.log")))

    print(f"{'policy':<14} {'L':>10} {'M':>10} {'S':>10} {'dur(s)':>8} {'migs':>6} {'compl':>5}")
    print("-" * 70)
    for p, info in rows:
        if not info.get("ok"):
            print(f"{p:<14} (no result: {info.get('error', 'missing')})")
            continue
        cells = []
        for cls in ("L", "M", "S"):
            s = info["per_class"].get(cls)
            if s is None:
                cells.append("    n/a   ")
            else:
                cells.append(f"{s['slo_met']:>2}/{s['n']:<2} {s['slo_rate']*100:>4.0f}%")
        migs = info.get("migrations", -1)
        migs_str = "n/a" if migs < 0 else str(migs)
        print(f"{p:<14} {cells[0]:>10} {cells[1]:>10} {cells[2]:>10} "
              f"{info['duration_s']:>8.1f} {migs_str:>6} {info['completed']:>5}")


if __name__ == "__main__":
    main()
