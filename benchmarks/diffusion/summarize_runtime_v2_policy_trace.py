#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Summarize runtime_v2 policy simulator traces by workload phase."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload-json", required=True)
    parser.add_argument("--trace", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workload = json.loads(Path(args.workload_json).read_text())["requests"]
    trace_events = json.loads(Path(args.trace).read_text())["traceEvents"]
    arrival_ms = {item["request_id"]: float(item["arrival_ms"]) for item in workload}

    finish_ms: dict[str, float] = {}
    dit_groups: set[tuple[str, int | None, int | None, str]] = set()
    for event in trace_events:
        if event.get("ph") != "X":
            continue
        event_args = event.get("args", {})
        request_id = event_args.get("request_id")
        if request_id not in arrival_ms:
            continue
        end_ms = float(event["ts"]) / 1000.0 + float(event["dur"]) / 1000.0
        finish_ms[request_id] = max(finish_ms.get(request_id, 0.0), end_ms)
        if event_args.get("kind") == "dit_step_chunk":
            dit_groups.add(
                (
                    request_id,
                    event_args.get("step_start"),
                    event_args.get("step_end"),
                    str(event_args.get("group_id")),
                )
            )

    by_phase: dict[str, list[float]] = {}
    groups_by_phase: dict[str, dict[str, int]] = {}
    for request_id, finish in finish_ms.items():
        phase = request_id.split("_", 1)[0]
        by_phase.setdefault(phase, []).append(finish - arrival_ms[request_id])
    for request_id, _step_start, _step_end, group_id in dit_groups:
        phase = request_id.split("_", 1)[0]
        group_counts = groups_by_phase.setdefault(phase, {})
        group_counts[group_id] = group_counts.get(group_id, 0) + 1

    for phase in sorted(by_phase):
        values = sorted(by_phase[phase])
        print(
            f"{phase}: n={len(values)} "
            f"avg_ms={statistics.mean(values):.3f} "
            f"p50_ms={statistics.median(values):.3f} "
            f"p95_ms={_percentile(values, 95):.3f} "
            f"p99_ms={_percentile(values, 99):.3f} "
            f"max_ms={max(values):.3f} "
            f"groups={groups_by_phase.get(phase, {})}"
        )
    all_values = sorted(value for values in by_phase.values() for value in values)
    if all_values:
        print(
            f"overall: n={len(all_values)} "
            f"avg_ms={statistics.mean(all_values):.3f} "
            f"p50_ms={statistics.median(all_values):.3f} "
            f"p95_ms={_percentile(all_values, 95):.3f} "
            f"p99_ms={_percentile(all_values, 99):.3f} "
            f"max_ms={max(all_values):.3f}"
        )


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    rank = (len(values) - 1) * percentile / 100.0
    lower = int(rank)
    upper = min(lower + 1, len(values) - 1)
    weight = rank - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


if __name__ == "__main__":
    main()
