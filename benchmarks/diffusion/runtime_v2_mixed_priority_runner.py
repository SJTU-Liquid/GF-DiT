#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""End-to-end runner for runtime_v2 mixed-priority video DiT experiments.

Builds an 8-rank topology with overlapping SP1/SP2/SP4/SP8 lanes, runs a
workload through one or more policies, and emits per-class metrics
(SLO attainment, latency percentiles, normalized latency, group switches,
reshard/preemption overhead) to a results JSON.

Example:

  python benchmarks/diffusion/runtime_v2_mixed_priority_runner.py \\
      --workload-json runtime_v2_mixed_priority.json \\
      --policy benchmarks.diffusion.runtime_v2_mixed_priority_policies:make_static_sp4 \\
              benchmarks.diffusion.runtime_v2_mixed_priority_policies:make_load_adaptive \\
              benchmarks.diffusion.runtime_v2_mixed_priority_policies:make_priority_load_adaptive \\
              benchmarks.diffusion.runtime_v2_mixed_priority_policies:make_step_preempt_pause \\
              benchmarks.diffusion.runtime_v2_mixed_priority_policies:make_step_preempt_demote \\
      --reshard-ms 50 \\
      --results-out results.json
"""

from __future__ import annotations

import argparse
import importlib
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

from benchmarks.diffusion.runtime_v2_policy_simulator import (
    CostModelStore,
    RuntimeV2PolicySimulator,
    SIM_TASK_LAUNCH_KIND,
    SimRequest,
    SimulationResult,
    _request_from_mapping,
)
from vllm_omni.diffusion.runtime_v2.adapters.common import STANDARD_TASK_KINDS
from vllm_omni.diffusion.runtime_v2.protocol import (
    ExecutionGroupSpec,
    ParallelSpec,
    TaskKind,
)
from vllm_omni.diffusion.runtime_v2.topology import RuntimeTopology, WorkerSpec


# 8-rank topology with one SP1 lane per rank, two SP4 lanes (disjoint),
# and one SP8 lane.  SP2 is intentionally omitted because the cost model
# shows SP2 is barely faster than SP1 — adding SP2 just inflates the
# decision space without giving the policy a useful Pareto point.
import itertools


def _build_topology_json(style: str, world_size: int = 8) -> list[dict[str, object]]:
    """Build an 8-rank topology with overlapping SP groups.

    Always emits:
      * 8 SP1 lanes (one per rank).
      * 1 SP8 lane covering all 8 ranks.

    The SP2 / SP4 enumeration depends on ``style``:

    * ``minimal``: only the 8 SP1 lanes and the SP8 group. Pair with
      ``--policy ...:make_edf_greedy``; that policy synthesizes
      intermediate SP-k groups dynamically via topology.ensure_group(),
      mirroring production. Other policies in this module require
      pre-declared SP2 / SP4 groups and will not work under ``minimal``.
    * ``partition``: the rigid {[0..3], [4..7]} SP4 partition, no SP2.
      Matches early experiments; useful to show the rank-overlap pathology.
    * ``sliding``: consecutive windows. SP2 = 7 pairs (i, i+1) for i in 0..6,
      SP4 = 5 windows (i..i+3) for i in 0..4. Models "ring topology with
      neighbor preference".
    * ``c84``: full combinatorial enumeration. SP2 = C(8,2)=28 pairs,
      SP4 = C(8,4)=70 subsets. Models a fully-connected NVLink switch
      where any rank subset can form an SP group. With rank-aware
      exclusion in the simulator, overlapping lanes serialize naturally.
    """
    groups: list[dict[str, object]] = [
        {"group_id": f"sp1_{i}", "ranks": [i], "tp": 1, "ulysses_degree": 1, "ring_degree": 1, "cfg": 1}
        for i in range(world_size)
    ]
    if style == "minimal":
        pass  # SP1 lanes + world group only; EdfGreedyPolicy fills in the rest.
    elif style == "partition":
        if world_size % 2 != 0:
            raise ValueError(f"partition style requires even world_size, got {world_size}")
        half = world_size // 2
        for i, ranks in enumerate((list(range(half)), list(range(half, world_size)))):
            groups.append(
                {
                    "group_id": f"sp{half}_{i}",
                    "ranks": list(ranks),
                    "tp": 1,
                    "ulysses_degree": half,
                    "ring_degree": 1,
                    "cfg": 1,
                }
            )
    elif style == "sliding":
        for i in range(world_size - 1):
            groups.append(
                {"group_id": f"sp2_{i}", "ranks": [i, i + 1], "tp": 1, "ulysses_degree": 2, "ring_degree": 1, "cfg": 1}
            )
        if world_size >= 4:
            for i in range(world_size - 3):
                groups.append(
                    {
                        "group_id": f"sp4_{i}",
                        "ranks": [i, i + 1, i + 2, i + 3],
                        "tp": 1,
                        "ulysses_degree": 4,
                        "ring_degree": 1,
                        "cfg": 1,
                    }
                )
    elif style == "c84":
        for idx, ranks in enumerate(itertools.combinations(range(world_size), 2)):
            groups.append(
                {"group_id": f"sp2_{idx}", "ranks": list(ranks), "tp": 1, "ulysses_degree": 2, "ring_degree": 1, "cfg": 1}
            )
        if world_size >= 4:
            for idx, ranks in enumerate(itertools.combinations(range(world_size), 4)):
                groups.append(
                    {"group_id": f"sp4_{idx}", "ranks": list(ranks), "tp": 1, "ulysses_degree": 4, "ring_degree": 1, "cfg": 1}
                )
    else:
        raise ValueError(f"unknown topology style {style!r}; use minimal, partition, sliding, c84")
    groups.append(
        {
            "group_id": f"sp{world_size}",
            "ranks": list(range(world_size)),
            "tp": 1,
            "ulysses_degree": world_size,
            "ring_degree": 1,
            "cfg": 1,
        }
    )
    return groups


DEFAULT_TOPOLOGY_JSON = _build_topology_json("sliding")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload-json", required=True)
    parser.add_argument("--cost-model-dir", default="cost-model/wan22-ti2v-5b")
    parser.add_argument(
        "--policy",
        nargs="+",
        required=True,
        help="one or more policy factory specs, e.g. module:make_xxx",
    )
    parser.add_argument(
        "--policy-name",
        nargs="+",
        default=None,
        help="optional display name per policy; defaults to factory name",
    )
    parser.add_argument(
        "--topology-json",
        default=None,
        help="override topology with a JSON list of group dicts. Takes precedence over --topology-style.",
    )
    parser.add_argument(
        "--topology-style",
        choices=("minimal", "partition", "sliding", "c84"),
        default="sliding",
        help=(
            "preset topology style. minimal=8 SP1 + SP8 only (pair with "
            "make_edf_greedy, which synthesizes SP2/SP4 dynamically), "
            "partition=rigid {[0..3],[4..7]}, sliding=consecutive windows "
            "(7 SP2 + 5 SP4), c84=full C(8,k) subsets (28 SP2 + 70 SP4). "
            "All variants include 8 SP1 lanes and 1 SP8 lane."
        ),
    )
    parser.add_argument(
        "--reshard-ms",
        type=float,
        default=3.0,
        help=(
            "simulated cost of cross-group state migration (ms). "
            "Matches runtime_v2_reshard_microbench typical p50."
        ),
    )
    parser.add_argument(
        "--reshard-samples-json",
        default=None,
        help=(
            "JSON list of real per-migration durations (ms). When set, each "
            "RESHARD samples its duration from these (seeded) instead of the "
            "flat --reshard-ms — replays the measured tail. Pass a single-value "
            "list to ablate the tail (flat mean)."
        ),
    )
    parser.add_argument(
        "--results-out",
        default=None,
        help="write metrics to this JSON path; if omitted, only printed to stdout",
    )
    parser.add_argument(
        "--trace-out-dir",
        default=None,
        help="if set, write per-policy chrome traces here",
    )
    parser.add_argument(
        "--launch-end-before-exec-end-ms",
        type=float,
        default=0.05,
    )
    parser.add_argument("--exec-start-delay-ms", type=float, default=0.005)
    parser.add_argument("--allowed-groups", default=None, help="comma-separated group_ids; restrict topology")
    parser.add_argument(
        "--world-size",
        type=int,
        default=8,
        help="Number of simulated worker ranks (default 8). Topology presets emit SP lanes up to this size.",
    )
    return parser.parse_args()


def _build_topology(
    topology_json_raw: str | None,
    allowed_groups: str | None,
    topology_style: str = "sliding",
    world_size: int = 8,
) -> RuntimeTopology:
    if topology_json_raw:
        topology_json = json.loads(topology_json_raw)
    else:
        topology_json = _build_topology_json(topology_style, world_size=world_size)
    if allowed_groups:
        allowed = {g.strip() for g in allowed_groups.split(",") if g.strip()}
        topology_json = [g for g in topology_json if g["group_id"] in allowed]
    workers = tuple(WorkerSpec(worker_rank=i, device_id=i) for i in range(world_size))
    groups: list[ExecutionGroupSpec] = []
    for raw in topology_json:
        ranks = tuple(int(r) for r in raw["ranks"])
        ulysses = int(raw.get("ulysses_degree", raw.get("ulysses", 1)))
        ring = int(raw.get("ring_degree", raw.get("ring", 1)))
        sp = ulysses * ring
        cfg = int(raw.get("cfg", 1))
        tp = int(raw.get("tp", len(ranks)))
        groups.append(
            ExecutionGroupSpec(
                group_id=raw["group_id"],
                ranks=ranks,
                parallel_spec=ParallelSpec(tp=tp, sp=sp, cfg=cfg),
                supported_task_kinds=STANDARD_TASK_KINDS,
                ulysses_degree=ulysses,
                ring_degree=ring,
            )
        )
    return RuntimeTopology(workers=workers, groups=tuple(groups))


def _load_factory(policy_spec: str):
    module_name, factory_name = policy_spec.split(":", 1)
    module = importlib.import_module(module_name)
    return getattr(module, factory_name)


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    s = sorted(values)
    rank = (len(s) - 1) * p / 100.0
    lower = int(rank)
    upper = min(lower + 1, len(s) - 1)
    weight = rank - lower
    return s[lower] * (1.0 - weight) + s[upper] * weight


def _request_finish_breakdown(
    result: SimulationResult,
    requests: list[SimRequest],
) -> dict[str, dict[str, Any]]:
    by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for req in requests:
        finish = result.request_finish_ms.get(req.request_id)
        if finish is None:
            continue
        latency_ms = finish - req.arrival_ms
        klass = (req.request_class or req.request_id.split("_", 1)[0]).upper()
        deadline_met = req.deadline_ms is None or finish <= req.deadline_ms
        norm_latency = None
        if req.profiled_optimal_latency_ms and req.profiled_optimal_latency_ms > 0:
            norm_latency = latency_ms / req.profiled_optimal_latency_ms
        by_class[klass].append(
            {
                "request_id": req.request_id,
                "arrival_ms": req.arrival_ms,
                "finish_ms": finish,
                "latency_ms": latency_ms,
                "deadline_ms": req.deadline_ms,
                "deadline_met": deadline_met,
                "normalized_latency": norm_latency,
            }
        )
    out: dict[str, dict[str, Any]] = {}
    for klass, items in sorted(by_class.items()):
        lat = [it["latency_ms"] for it in items]
        met = sum(1 for it in items if it["deadline_met"])
        norm = [it["normalized_latency"] for it in items if it["normalized_latency"] is not None]
        out[klass] = {
            "n": len(items),
            "completed": len(items),
            "slo_attained": met,
            "slo_attainment_ratio": met / len(items) if items else 0.0,
            "avg_latency_ms": statistics.mean(lat) if lat else 0.0,
            "p50_latency_ms": _percentile(lat, 50),
            "p95_latency_ms": _percentile(lat, 95),
            "p99_latency_ms": _percentile(lat, 99),
            "max_latency_ms": max(lat) if lat else 0.0,
            "avg_normalized_latency": statistics.mean(norm) if norm else 0.0,
            "p95_normalized_latency": _percentile(norm, 95) if norm else 0.0,
        }
    return out


def _timeline_breakdown(result: SimulationResult, requests: list[SimRequest]) -> dict[str, Any]:
    """Reshard / preemption / per-stage / group-switch statistics."""
    request_classes = {
        req.request_id: (req.request_class or req.request_id.split("_", 1)[0]).upper()
        for req in requests
    }
    reshard_count = 0
    reshard_time_ms = 0.0
    stage_time_ms: dict[str, float] = defaultdict(float)
    dit_groups_per_req: dict[str, set[str]] = defaultdict(set)
    dit_groups_per_class: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for event in result.timeline:
        if event.kind == SIM_TASK_LAUNCH_KIND:
            continue
        duration_ms = max(0.0, event.end_ms - event.start_ms)
        stage_time_ms[event.kind] += duration_ms
        if event.kind == TaskKind.RESHARD.value:
            reshard_count += 1
            reshard_time_ms += duration_ms
            continue
        if event.kind == TaskKind.DIT_STEP_CHUNK.value:
            dit_groups_per_req[event.request_id].add(event.group_id)
            klass = request_classes.get(event.request_id, "?")
            dit_groups_per_class[klass][event.group_id] += 1
    group_switch_count = sum(max(0, len(groups) - 1) for groups in dit_groups_per_req.values())
    return {
        "reshard_count": reshard_count,
        "reshard_time_ms": reshard_time_ms,
        "stage_time_ms": dict(stage_time_ms),
        "group_switch_total": group_switch_count,
        "requests_with_switch": sum(1 for groups in dit_groups_per_req.values() if len(groups) > 1),
        "dit_groups_per_class": {klass: dict(g) for klass, g in dit_groups_per_class.items()},
    }


def _wait_breakdown(result: SimulationResult, requests: list[SimRequest]) -> dict[str, dict[str, float]]:
    """Time-to-first-DiT step for foreground requests, by class."""
    first_dit_ms: dict[str, float] = {}
    for event in result.timeline:
        if event.kind != TaskKind.DIT_STEP_CHUNK.value:
            continue
        existing = first_dit_ms.get(event.request_id)
        if existing is None or event.start_ms < existing:
            first_dit_ms[event.request_id] = event.start_ms
    by_class: dict[str, list[float]] = defaultdict(list)
    for req in requests:
        first = first_dit_ms.get(req.request_id)
        if first is None:
            continue
        wait = max(0.0, first - req.arrival_ms)
        klass = (req.request_class or req.request_id.split("_", 1)[0]).upper()
        by_class[klass].append(wait)
    out: dict[str, dict[str, float]] = {}
    for klass, values in sorted(by_class.items()):
        out[klass] = {
            "avg_wait_to_first_dit_ms": statistics.mean(values) if values else 0.0,
            "p95_wait_to_first_dit_ms": _percentile(values, 95),
        }
    return out


def _run_one_policy(
    policy_spec: str,
    name: str,
    requests: list[SimRequest],
    topology: RuntimeTopology,
    cost_models: CostModelStore,
    args: argparse.Namespace,
) -> dict[str, Any]:
    factory = _load_factory(policy_spec)
    simulator = RuntimeV2PolicySimulator(
        topology=topology,
        policy_factory=factory,
        cost_models=cost_models,
        reshard_ms=args.reshard_ms,
        reshard_samples=(
            json.load(open(args.reshard_samples_json))
            if getattr(args, "reshard_samples_json", None) else None
        ),
        launch_end_before_exec_end_ms=args.launch_end_before_exec_end_ms,
        exec_start_delay_ms=args.exec_start_delay_ms,
    )
    result = simulator.run(requests)
    breakdown = _request_finish_breakdown(result, requests)
    timeline = _timeline_breakdown(result, requests)
    waits = _wait_breakdown(result, requests)
    makespan = max(result.request_finish_ms.values(), default=0.0)
    arrival_min = min(result.request_arrival_ms.values(), default=0.0)
    completed = sum(stats["n"] for stats in breakdown.values())

    # Overall summary
    overall_latencies = [
        item["latency_ms"]
        for stats in breakdown.values()
        for item in []  # filled below
    ]
    overall_latencies = []
    for req in requests:
        finish = result.request_finish_ms.get(req.request_id)
        if finish is not None:
            overall_latencies.append(finish - req.arrival_ms)
    overall = {
        "completed_requests": completed,
        "total_requests": len(requests),
        "makespan_ms": makespan,
        "duration_ms": makespan - arrival_min,
        "global_idle_ms": result.global_idle_ms,
        "global_idle_frac": result.global_idle_frac,
        "avg_latency_ms": statistics.mean(overall_latencies) if overall_latencies else 0.0,
        "p95_latency_ms": _percentile(overall_latencies, 95),
    }
    per_request: list[dict[str, Any]] = []
    for req in requests:
        finish = result.request_finish_ms.get(req.request_id)
        if finish is None:
            continue
        klass = (req.request_class or req.request_id.split("_", 1)[0]).upper()
        lateness_ms = (
            (finish - req.deadline_ms) if req.deadline_ms is not None else None
        )
        per_request.append(
            {
                "request_id": req.request_id,
                "request_class": klass,
                "latency_ms": finish - req.arrival_ms,
                "deadline_ms": req.deadline_ms,
                "deadline_met": req.deadline_ms is None or finish <= req.deadline_ms,
                "lateness_ms": lateness_ms,
            }
        )
    out = {
        "policy": policy_spec,
        "name": name,
        "overall": overall,
        "by_class": breakdown,
        "timeline": timeline,
        "wait_to_first_dit": waits,
        "per_request": per_request,
    }
    if args.trace_out_dir:
        path = Path(args.trace_out_dir) / f"trace_{name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"traceEvents": result.trace_events}))
        out["trace_path"] = str(path)
    return out


def _print_table(results: list[dict[str, Any]]) -> None:
    header = (
        f"{'policy':<26} {'class':<6} {'n':>4} {'SAR':>6} "
        f"{'avg(s)':>9} {'p95(s)':>9} {'p99(s)':>9} "
        f"{'norm_p95':>9} {'wait_p95(s)':>11}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        for klass in sorted(r["by_class"].keys()):
            stats = r["by_class"][klass]
            wait = r["wait_to_first_dit"].get(klass, {}).get("p95_wait_to_first_dit_ms", 0.0)
            print(
                f"{r['name'][:26]:<26} {klass:<6} {stats['n']:>4} "
                f"{stats['slo_attainment_ratio']*100:>5.1f}% "
                f"{stats['avg_latency_ms']/1000:>9.2f} "
                f"{stats['p95_latency_ms']/1000:>9.2f} "
                f"{stats['p99_latency_ms']/1000:>9.2f} "
                f"{stats['p95_normalized_latency']:>9.2f} "
                f"{wait/1000:>11.2f}"
            )
        ts = r["timeline"]
        ov = r["overall"]
        print(
            f"  {r['name'][:24]:<24} overall: "
            f"makespan={ov['makespan_ms']/1000:.1f}s "
            f"completed={ov['completed_requests']}/{ov['total_requests']} "
            f"idle={ov['global_idle_frac']*100:.1f}% "
            f"reshards={ts['reshard_count']} "
            f"switches={ts['group_switch_total']}"
        )
        print()


def main() -> None:
    args = parse_args()
    workload_data = json.loads(Path(args.workload_json).read_text())
    requests = [_request_from_mapping(item) for item in workload_data.get("requests", [])]
    if not requests:
        print(f"no requests in {args.workload_json}", file=sys.stderr)
        sys.exit(1)
    topology = _build_topology(
        args.topology_json, args.allowed_groups, args.topology_style, world_size=args.world_size
    )
    cost_models = CostModelStore.load_dir(args.cost_model_dir)

    names = list(args.policy_name) if args.policy_name else []
    while len(names) < len(args.policy):
        names.append(args.policy[len(names)].rsplit(":", 1)[-1])

    results: list[dict[str, Any]] = []
    for policy_spec, name in zip(args.policy, names):
        print(f"[runner] policy={name} spec={policy_spec}")
        try:
            results.append(_run_one_policy(policy_spec, name, requests, topology, cost_models, args))
        except Exception as exc:
            print(f"[runner] policy={name} failed: {exc!r}", file=sys.stderr)
            results.append(
                {
                    "policy": policy_spec,
                    "name": name,
                    "error": repr(exc),
                }
            )

    successful = [r for r in results if "error" not in r]
    if successful:
        _print_table(successful)
    failed = [r for r in results if "error" in r]
    for r in failed:
        print(f"[runner] FAILED {r['name']}: {r['error']}", file=sys.stderr)

    if args.results_out:
        Path(args.results_out).write_text(json.dumps({"results": results, "args": vars(args)}, indent=2))
        print(f"wrote {args.results_out}")


if __name__ == "__main__":
    main()
