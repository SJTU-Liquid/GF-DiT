#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Analyze a runtime_v2 edf_best_fit stress-test server log.

Parses a captured `vllm serve ... --runtime-v2-scheduler-policy edf_best_fit`
log and reports:

  * noise breakdown (cpu-thread-trace spam vs. real lines)
  * per-request lifecycle (submit shape, plan compile, completion)
  * DiT chunk SP-size distribution -- the edf_best_fit "downshift" signal:
    if every chunk runs at SP=max, best-fit collapsed to greedy largest-fit,
    which means the workload never created enough contention to test it
  * exec_only_ms stats per SP size
  * implicit-migration (reshard) count

Usage:
  .venv/bin/python scripts/analyze_edf_stress_log.py edf-best-stress-4.log
  .venv/bin/python scripts/analyze_edf_stress_log.py edf-best-stress-4.log --per-request
"""

from __future__ import annotations

import argparse
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path


# --- line patterns ----------------------------------------------------------
RE_SUBMIT = re.compile(
    r"runtime_v2 submit: request_id=(?P<rid>\S+).*?"
    r"chunk=(?P<chunk>\d+) steps=(?P<steps>\d+) frames=(?P<frames>\d+) "
    r"size=(?P<w>\d+)x(?P<h>\d+)"
)
RE_PLAN = re.compile(
    r"runtime_v2 plan compiled: request_id=(?P<rid>\S+) tasks=(?P<tasks>\d+).*?"
    r"'num_steps': (?P<steps>\d+).*?'latent_seq_len': (?P<lsl>\d+)"
)
RE_DIT = re.compile(
    r"runtime_v2 worker dit chunk timing: rank=(?P<rank>\d+) "
    r"group=(?P<group>\S+) task_id=(?P<rid>[^:]+):dit_step_chunk:(?P<chunk>\d+) "
    r"exec_(?:gpu|only)_ms=(?P<exec>[\d.]+) (?:exec_cpu_ms=[\d.]+ )?"
    r"session_activate_ms=(?P<act>[\d.]+)"
)
RE_MIGRATE = re.compile(
    r"runtime_v2 implicit migration: request_id=(?P<rid>\S+).*?"
    r"src_group=(?P<src>\S+) dst_group=(?P<dst>\S+)"
)
RE_TS = re.compile(r"\b(\d\d:\d\d:\d\d)\b")


def sp_of_group(group: str) -> int:
    """Derive SP degree from a runtime_v2 execution-group id."""
    if "world" in group:
        return 4  # static full-world group
    m = re.search(r"sp(\d+)", group)
    if m:
        return int(m.group(1))
    return 1


def pct(num: int, den: int) -> str:
    return f"{(100.0 * num / den):.1f}%" if den else "0.0%"


def fmt_stats(values: list[float]) -> str:
    if not values:
        return "n=0"
    s = sorted(values)
    p95 = s[min(len(s) - 1, int(0.95 * len(s)))]
    return (
        f"n={len(s)} mean={statistics.mean(s):.0f} "
        f"p50={statistics.median(s):.0f} p95={p95:.0f} "
        f"min={s[0]:.0f} max={s[-1]:.0f}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("log", type=Path)
    ap.add_argument("--per-request", action="store_true",
                    help="dump one line per request with its SP-size mix")
    args = ap.parse_args()

    total_lines = 0
    noise_trace = 0
    noise_idle = 0
    other_noise = 0  # FutureWarning / RuntimeWarning / venv lines

    submits: dict[str, dict] = {}
    plans: dict[str, dict] = {}
    # per request: Counter of sp -> #chunks, list of exec_ms
    req_sp: dict[str, Counter] = defaultdict(Counter)
    req_exec: dict[str, list[float]] = defaultdict(list)
    exec_by_sp: dict[int, list[float]] = defaultdict(list)
    migrations = 0
    first_ts = last_ts = None

    with args.log.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            total_lines += 1

            mts = RE_TS.search(line)
            if mts:
                if first_ts is None:
                    first_ts = mts.group(1)
                last_ts = mts.group(1)

            if "cpu thread trace" in line:
                noise_trace += 1
                if "action=idle_sleep" in line:
                    noise_idle += 1
                continue
            if ("FutureWarning" in line or "RuntimeWarning" in line
                    or "warnings.warn" in line or "site-packages" in line):
                other_noise += 1
                continue

            m = RE_SUBMIT.search(line)
            if m:
                submits[m.group("rid")] = {
                    "chunk": int(m.group("chunk")),
                    "steps": int(m.group("steps")),
                    "frames": int(m.group("frames")),
                    "size": f"{m.group('w')}x{m.group('h')}",
                }
                continue
            m = RE_PLAN.search(line)
            if m:
                plans[m.group("rid")] = {
                    "tasks": int(m.group("tasks")),
                    "steps": int(m.group("steps")),
                    "latent_seq_len": int(m.group("lsl")),
                }
                continue
            m = RE_DIT.search(line)
            if m:
                # one log line per rank; count the chunk once (rank 0) for
                # chunk-level stats, but exec_ms is per-rank wall time so
                # collect them all and they agree within a few ms.
                rid = m.group("rid")
                sp = sp_of_group(m.group("group"))
                exec_ms = float(m.group("exec"))
                if m.group("rank") == "0" or sp == 1:
                    req_sp[rid][sp] += 1
                req_exec[rid].append(exec_ms)
                exec_by_sp[sp].append(exec_ms)
                continue
            m = RE_MIGRATE.search(line)
            if m:
                migrations += 1
                continue

    real_lines = total_lines - noise_trace - other_noise

    print("=" * 64)
    print(f"log: {args.log}   span: {first_ts} -> {last_ts}")
    print("=" * 64)
    print("\n## line breakdown")
    print(f"  total lines        : {total_lines}")
    print(f"  cpu-thread-trace   : {noise_trace:>7}  ({pct(noise_trace, total_lines)})")
    print(f"    of which idle    : {noise_idle:>7}  ({pct(noise_idle, total_lines)})")
    print(f"  dep warnings/venv  : {other_noise:>7}  ({pct(other_noise, total_lines)})")
    print(f"  real signal lines  : {real_lines:>7}  ({pct(real_lines, total_lines)})")
    if noise_idle > real_lines:
        print("  >> idle_sleep noise outnumbers signal: control loop was mostly")
        print("     idle -> the workload did NOT keep the server busy.")

    # request set: union of submit + plan + dit
    all_rids = set(submits) | set(plans) | set(req_sp)
    # warmup request has size like 1024x1024 steps=1 -- flag separately
    warmups = {r for r, s in submits.items() if s.get("steps", 0) <= 2}
    real_rids = sorted(all_rids - warmups)

    print("\n## requests")
    print(f"  total request ids  : {len(all_rids)}  (warmup-ish: {len(warmups)})")
    print(f"  real requests      : {len(real_rids)}")
    print(f"  implicit migrations: {migrations}  (reshards at stage boundaries)")

    # SP-size distribution across all DiT chunks
    sp_chunk_total: Counter = Counter()
    for rid, spc in req_sp.items():
        sp_chunk_total.update(spc)
    total_chunks = sum(sp_chunk_total.values())
    print("\n## DiT chunk SP-size distribution  (edf_best_fit downshift signal)")
    if total_chunks == 0:
        print("  no dit chunk timing lines found")
    else:
        for sp in sorted(sp_chunk_total):
            c = sp_chunk_total[sp]
            print(f"  SP={sp:<2}: {c:>7} chunks  ({pct(c, total_chunks)})")
        max_sp = max(sp_chunk_total)
        frac_max = sp_chunk_total[max_sp] / total_chunks
        if frac_max > 0.95:
            print(f"  >> {pct(sp_chunk_total[max_sp], total_chunks)} of chunks at SP={max_sp}: "
                  "best-fit collapsed to greedy largest-fit.")
            print("     The workload never created concurrent deadline pressure,")
            print("     so PASS-2 redistribute promoted everything to max SP.")
        else:
            print(f"  >> best-fit DID downshift {pct(total_chunks - sp_chunk_total[max_sp], total_chunks)} "
                  "of chunks below max SP -- some real contention occurred.")

    print("\n## exec_only_ms per SP size")
    for sp in sorted(exec_by_sp):
        print(f"  SP={sp:<2}: {fmt_stats(exec_by_sp[sp])}")

    # how many real requests ran mixed SP within themselves (pass-2 / reshard)
    mixed = [r for r in real_rids if len(req_sp.get(r, {})) > 1]
    print(f"\n## requests that ran mixed SP sizes: {len(mixed)} / {len(real_rids)}")
    if mixed and not args.per_request:
        print("  (use --per-request to see the per-request SP mix)")

    if args.per_request:
        print("\n## per-request SP mix")
        hdr = f"  {'request_id':<40} {'steps':>5} {'lat_seq':>8}  sp_mix(chunks)  exec_ms"
        print(hdr)
        for rid in real_rids:
            spc = req_sp.get(rid, Counter())
            mix = " ".join(f"sp{sp}:{spc[sp]}" for sp in sorted(spc)) or "-"
            plan = plans.get(rid, {})
            ex = req_exec.get(rid, [])
            exs = f"mean={statistics.mean(ex):.0f}" if ex else "-"
            print(f"  {rid[:40]:<40} {plan.get('steps', '-'):>5} "
                  f"{plan.get('latent_seq_len', '-'):>8}  {mix:<22} {exs}")

    print("\nnote: per-request end-to-end latency / SLO attainment lives in the")
    print("bench client --output-file JSON, not this server log. This script")
    print("characterizes scheduler behavior (SP placement), not SLO outcomes.")


if __name__ == "__main__":
    main()
