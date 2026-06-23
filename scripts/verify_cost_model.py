#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Verify runtime_v2 cost-model predictions against a captured server log,
and reconstruct GPU bubbles (idle time) from DiT chunk timings.

Two reports:

  1. COST MODEL  -- per (request class, SP, cost stage) actual exec_gpu_ms (from
     `worker dit chunk timing` log lines) vs the cost-model quadratic
     prediction a*x^2 + b*x + c at the request's latent_seq_len. Flags
     when the workload's latent_seq_len is outside the cost-model's
     fitted sample range (extrapolation = unreliable prediction).

  2. BUBBLES     -- per-rank busy time reconstructed from dit chunk
     exec_gpu_ms, the global wall span, GPU utilization, and the gap
     structure (inter-request idle vs intra-request idle).

Usage:
  .venv/bin/python scripts/verify_cost_model.py edf-best-stress-4.log \
      --cost-model-dir cost-model/wan22-ti2v-5b-fullrange
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

RE_PLAN = re.compile(
    r"runtime_v2 plan compiled: request_id=(?P<rid>\S+) tasks=\d+.*?"
    r"'num_steps': (?P<steps>\d+).*?'latent_seq_len': (?P<lsl>\d+).*?"
    r"'height': (?P<h>\d+), 'width': (?P<w>\d+), 'num_frames': (?P<nf>\d+)"
)
RE_DIT = re.compile(
    r"(?P<ts>\d\d:\d\d:\d\d) \[multiproc_worker\.py:\d+\] "
    r"runtime_v2 worker dit chunk timing: rank=(?P<rank>\d+) "
    r"group=(?P<group>\S+) task_id=(?P<rid>[^:]+):dit_step_chunk:(?P<chunk>\d+) "
    # exec_gpu_ms (CUDA-event timed) on current logs; exec_only_ms on old ones.
    r"exec_(?:gpu|only)_ms=(?P<exec>[\d.]+)"
    r"(?:.*?cost_model_stage=(?P<stage>\S+))?"
    r"(?:.*?do_true_cfg=(?P<do_true_cfg>\S+))?"
)

# num_steps -> request class (from generate_runtime_v2_mixed_priority_workload)
STEPS_TO_CLASS = {25: "S", 50: "M", 100: "L"}


def sp_of_group(group: str) -> int:
    if "world" in group:
        return 4
    m = re.search(r"sp(\d+)", group)
    return int(m.group(1)) if m else 1


def ts_to_s(ts: str) -> int:
    h, m, s = (int(x) for x in ts.split(":"))
    return h * 3600 + m * 60 + s


DIT_COST_STAGES = ("dit_step_chunk", "dit_step_chunk_cfg", "dit_step_chunk_nocfg")


def _resolve_stage(stages: dict, stage_name: str) -> tuple[dict, str, float]:
    if stage_name in stages:
        return stages[stage_name], stage_name, 1.0
    if stage_name == "dit_step_chunk":
        if "dit_step_chunk_cfg" in stages:
            return stages["dit_step_chunk_cfg"], "dit_step_chunk_cfg", 1.0
    if stage_name == "dit_step_chunk_cfg":
        if "dit_step_chunk" in stages:
            return stages["dit_step_chunk"], "dit_step_chunk", 1.0
    if stage_name == "dit_step_chunk_nocfg":
        if "dit_step_chunk_cfg" in stages:
            return stages["dit_step_chunk_cfg"], "dit_step_chunk_cfg", 0.5
        if "dit_step_chunk" in stages:
            return stages["dit_step_chunk"], "dit_step_chunk", 0.5
    raise KeyError(stage_name)


def load_cost_model(cm_dir: Path) -> dict[int, dict]:
    """sp -> {'stages': {stage -> coeffs/x-range}} for DiT chunk stages."""
    out: dict[int, dict] = {}
    for path in sorted(cm_dir.glob("tp1_ulysses*_ring1_cfg1.json")):
        doc = json.loads(path.read_text())
        sp = int(doc["parallelism"]["sp"])
        stages: dict[str, dict] = {}
        for stage_name in DIT_COST_STAGES:
            stage, source_stage, scale = _resolve_stage(doc["stages"], stage_name)
            xs = [s[0] for s in stage.get("samples", [])]
            stages[stage_name] = {
                "coeffs": stage["coeffs"],
                "xmin": min(xs) if xs else float("-inf"),
                "xmax": max(xs) if xs else float("inf"),
                "r2": stage.get("r2"),
                "source_stage": source_stage,
                "scale": scale,
            }
        out[sp] = {"stages": stages}
    return out


def predict(stage_model: dict, x: float) -> float:
    coeffs = stage_model["coeffs"]
    return (coeffs["a"] * x * x + coeffs["b"] * x + coeffs["c"]) * stage_model.get("scale", 1.0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("log", type=Path)
    ap.add_argument("--cost-model-dir", type=Path,
                    default=Path("cost-model/wan22-ti2v-5b-fullrange"))
    args = ap.parse_args()

    cost_model = load_cost_model(args.cost_model_dir)
    plans: dict[str, dict] = {}
    # (class, sp, cost_model_stage) -> list of exec_only_ms
    actual: dict[tuple[str, int, str], list[float]] = {}
    # rank -> total busy ms ; rank -> list of (start_s, end_s)
    rank_busy_ms: dict[int, float] = {}
    # request -> (first dit end_s, last dit end_s)
    req_window: dict[str, list[int]] = {}
    first_s = last_s = None

    with args.log.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = RE_PLAN.search(line)
            if m:
                plans[m.group("rid")] = {
                    "steps": int(m.group("steps")),
                    "lsl": int(m.group("lsl")),
                    "voxels": int(m.group("h")) * int(m.group("w")) * int(m.group("nf")),
                }
                continue
            m = RE_DIT.search(line)
            if m:
                rid = m.group("rid")
                sp = sp_of_group(m.group("group"))
                rank = int(m.group("rank"))
                exec_ms = float(m.group("exec"))
                stage = m.group("stage") or "dit_step_chunk"
                end_s = ts_to_s(m.group("ts"))
                first_s = end_s if first_s is None else min(first_s, end_s)
                last_s = end_s if last_s is None else max(last_s, end_s)
                rank_busy_ms[rank] = rank_busy_ms.get(rank, 0.0) + exec_ms
                plan = plans.get(rid)
                klass = STEPS_TO_CLASS.get(plan["steps"], "?") if plan else "?"
                actual.setdefault((klass, sp, stage), []).append(exec_ms)
                w = req_window.setdefault(rid, [end_s, end_s])
                w[0] = min(w[0], end_s)
                w[1] = max(w[1], end_s)

    # ---- report 1: cost model vs actual --------------------------------
    print("=" * 74)
    print("COST MODEL  vs  ACTUAL   (dit_step_chunk exec_gpu_ms)")
    print("=" * 74)
    # per-class latent_seq_len (should be one value per class)
    class_lsl: dict[str, set] = {}
    for plan in plans.values():
        klass = STEPS_TO_CLASS.get(plan["steps"], "?")
        class_lsl.setdefault(klass, set()).add(plan["lsl"])
    print("\nworkload latent_seq_len per class:")
    for klass in ("S", "M", "L"):
        if klass in class_lsl:
            print(f"  {klass}: {sorted(class_lsl[klass])}")
    fitted = next(iter(cost_model.values()))
    print("\ncost-model fitted latent_seq_len ranges:")
    for stage_name in DIT_COST_STAGES:
        stage_model = fitted["stages"][stage_name]
        source = stage_model["source_stage"]
        scale = stage_model["scale"]
        suffix = "" if source == stage_name and scale == 1.0 else f" (from {source} x{scale:g})"
        print(
            f"  {stage_name}: [{stage_model['xmin']:.0f}, {stage_model['xmax']:.0f}]{suffix}"
        )
    print(f"\n{'class':<6}{'sp':<4}{'stage':<23}{'lat_seq':>9}{'n':>7}"
          f"{'actual_p50':>12}{'predicted':>11}{'act/pred':>10}  range")
    for klass in ("S", "M", "L"):
        lsls = sorted(class_lsl.get(klass, []))
        if not lsls:
            continue
        x = lsls[0]
        for sp in sorted(cost_model):
            for stage_name in DIT_COST_STAGES:
                key = (klass, sp, stage_name)
                if key not in actual:
                    continue
                vals = actual[key]
                p50 = statistics.median(vals)
                cm = cost_model[sp]["stages"][stage_name]
                pred = predict(cm, x)
                ratio = p50 / pred if pred else float("nan")
                in_range = "ok" if cm["xmin"] <= x <= cm["xmax"] else "EXTRAPOLATED"
                print(f"{klass:<6}{sp:<4}{stage_name:<23}{x:>9.0f}{len(vals):>7}"
                      f"{p50:>12.1f}{pred:>11.1f}{ratio:>9.2f}x  {in_range}")
    print("\n  act/pred ~1.0 = model matches. Far from 1.0 with EXTRAPOLATED")
    print("  means the workload runs outside the cost-model's fitted domain;")
    print("  the policy's finish-time math is then unreliable.")

    # ---- report 2: bubbles ---------------------------------------------
    print("\n" + "=" * 74)
    print("BUBBLES  (GPU idle reconstructed from dit chunk exec)")
    print("=" * 74)
    if first_s is None:
        print("no dit chunk timing found")
        return
    span_s = max(1, last_s - first_s)
    n_ranks = max(rank_busy_ms) + 1 if rank_busy_ms else 1
    total_busy_s = sum(rank_busy_ms.values()) / 1000.0
    capacity_s = n_ranks * span_s
    util = total_busy_s / capacity_s if capacity_s else 0.0
    print(f"\nwall span            : {span_s} s  ({span_s/60:.1f} min)")
    print(f"ranks                : {n_ranks}")
    print(f"total DiT busy        : {total_busy_s:.0f} rank-s")
    print(f"cluster capacity     : {capacity_s:.0f} rank-s")
    print(f"DiT GPU utilization  : {util*100:.1f}%   -> bubble = {(1-util)*100:.1f}%")
    print("\nper-rank DiT busy:")
    for rank in sorted(rank_busy_ms):
        busy_s = rank_busy_ms[rank] / 1000.0
        print(f"  rank {rank}: {busy_s:>7.0f} s busy  "
              f"({busy_s/span_s*100:.1f}% of wall)")

    # inter-request idle: sort request windows, sum gaps between them
    windows = sorted(req_window.values())
    inter_gap = 0
    covered_end = windows[0][1] if windows else 0
    union = 0
    for start, end in windows:
        if start > covered_end:
            inter_gap += start - covered_end
            covered_end = end
        else:
            covered_end = max(covered_end, end)
    # union of request-active spans
    cur_s, cur_e = windows[0]
    for start, end in windows[1:]:
        if start > cur_e:
            union += cur_e - cur_s
            cur_s, cur_e = start, end
        else:
            cur_e = max(cur_e, end)
    union += cur_e - cur_s
    print(f"\nrequests active span : {union} s  "
          f"({union/span_s*100:.0f}% of wall) -- at least one request running")
    print(f"inter-request idle   : {span_s - union} s  "
          f"({(span_s-union)/span_s*100:.0f}% of wall) -- ZERO requests running")
    print(f"intra-request bubble : {union - total_busy_s/n_ranks:.0f} s-equiv  "
          "-- requests running but not all ranks fed")
    print("\n  inter-request idle  = workload too sparse (raise --rate-scale).")
    print("  intra-request bubble = serial stage chain (text->dit->vae),")
    print("  single-rank aux stages idling 3 ranks, and reshard stalls.")


if __name__ == "__main__":
    main()
