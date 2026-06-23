#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate placement-agnostic runtime_v2 simulator workloads.

The default scenario has three phases:
1. isolated requests, where a latency policy should prefer high SP
2. a high-rate surge, where a throughput policy should prefer lower SP lanes
3. isolated cooldown requests, where the policy can scale back up
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="runtime_v2_elastic_workload.json")
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--num-frames", type=int, default=81)
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--denoise-chunk-size", type=int, default=1)
    parser.add_argument("--text-seq-len", type=int, default=512)
    parser.add_argument("--low-requests", type=int, default=2)
    parser.add_argument("--low-gap-ms", type=float, default=90_000.0)
    parser.add_argument("--surge-requests", type=int, default=16)
    parser.add_argument("--surge-interval-ms", type=float, default=500.0)
    parser.add_argument("--surge-start-ms", type=float, default=None)
    parser.add_argument("--cooldown-requests", type=int, default=2)
    parser.add_argument("--cooldown-gap-ms", type=float, default=360_000.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    requests: list[dict[str, int | float | str]] = []
    request_idx = 0

    def add_request(arrival_ms: float, phase: str) -> None:
        nonlocal request_idx
        requests.append(
            {
                "request_id": f"{phase}_{request_idx:04d}",
                "arrival_ms": round(float(arrival_ms), 3),
                "height": args.height,
                "width": args.width,
                "num_frames": args.num_frames,
                "num_steps": args.num_steps,
                "denoise_chunk_size": args.denoise_chunk_size,
                "text_seq_len": args.text_seq_len,
            }
        )
        request_idx += 1

    for idx in range(args.low_requests):
        add_request(idx * args.low_gap_ms, "low")

    surge_start_ms = args.surge_start_ms
    if surge_start_ms is None:
        surge_start_ms = max(args.low_requests, 1) * args.low_gap_ms
    for idx in range(args.surge_requests):
        add_request(surge_start_ms + idx * args.surge_interval_ms, "surge")

    cooldown_start_ms = surge_start_ms + args.surge_requests * args.surge_interval_ms + args.cooldown_gap_ms
    for idx in range(args.cooldown_requests):
        add_request(cooldown_start_ms + idx * args.cooldown_gap_ms, "cooldown")

    workload = {
        "description": "Elastic SP workload: isolated latency phase, high-rate surge, cooldown.",
        "requests": requests,
    }
    Path(args.out).write_text(json.dumps(workload, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.out} requests={len(requests)}")


if __name__ == "__main__":
    main()
