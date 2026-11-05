#!/usr/bin/env python3
"""Rebuild Table 1 (NCCL subgroup setup cost) and the GFC group-registration cost.

Inputs (raw benchmark outputs):
  logs/nccl_subgroup_cost_h20_sizesweep.json  benchmarks/distributed/nccl_subgroup_cost.py, 8x H20
  logs/gfc_group_cost_ws2.json                benchmarks/distributed/gfc_group_cost.py, 2x H20
Writes figs/table1_group_setup.md.  Run from ae/figures:  python3 scripts/summarize_group_setup.py
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NCCL = ROOT / "logs" / "nccl_subgroup_cost_h20_sizesweep.json"
GFC = ROOT / "logs" / "gfc_group_cost_ws2.json"
OUT = ROOT / "figs" / "table1_group_setup.md"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nccl", type=Path, default=NCCL)
    ap.add_argument("--gfc", type=Path, default=GFC)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    nccl = json.loads(args.nccl.read_text())
    lines = [f"Table 1: NCCL subgroup setup ({nccl['world']}x {nccl['device_name']}, torch {nccl['torch']}, "
             f"NCCL {nccl['nccl']})", "",
             "| Group size | new_group (ms) | First coll. (ms) | Warm coll. (ms) | Memory (MB/GPU) |",
             "|---:|---:|---:|---:|---:|"]
    for r in nccl["size_sweep"]:
        lines.append(f"| {r['size']} | {r['newgroup_cpu_ms_max']:.2f} | {r['first_coll_ms_mean']:.1f} | "
                     f"{r['warm_coll_ms_max']:.2f} | {r['mem_free_drop_mb_mean']:.0f} |")

    gfc = json.loads(args.gfc.read_text())
    reg_us = [c["register_ms_max"] * 1000 for c in gfc["churn"]]
    mem = gfc["churn"][-1]["cumulative_drop_mb"]
    lines += ["", f"GFC group registration ({gfc['world']}x {gfc['device_name']}, {len(reg_us)} successive "
              f"size-{gfc['churn_size']} groups):",
              f"- per-group register time: median {statistics.median(reg_us):.1f} us "
              f"(min {min(reg_us):.1f}, first/cold {reg_us[0]:.1f}); paper: ~60 us",
              f"- per-group memory: {mem:.0f} MB after {len(reg_us)} groups (NCCL: ~485-535 MB each)",
              f"- one-time runtime init: {gfc['runtime_init_ms']:.0f} ms, {gfc['runtime_init_mem_mb']:.0f} MB "
              f"shared symmetric region"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
