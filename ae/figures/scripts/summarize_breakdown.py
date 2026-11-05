#!/usr/bin/env python3
"""Rebuild Table 2 (policy breakdown) and the 40%-load numbers of Sec. 6.4.

Reads the real 4-GPU H20 Wan2.2 runs collected for the breakdown
(logs/rebuttal_real_h20/*.tgz; launch scripts alongside) and writes
figs/table2_breakdown.{csv,md}.  Run from ae/figures:  python3 scripts/summarize_breakdown.py
"""
from __future__ import annotations

import csv
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "logs" / "rebuttal_real_h20" / "breakdown_results.tgz"
LOAD40 = ROOT / "logs" / "rebuttal_real_h20" / "breakdown_load40_results.tgz"
OUT = ROOT / "figs" / "table2_breakdown"

# (configuration, added mechanism, bundle, member)
ROWS = [
    ("Legacy (SP4)", "---", MAIN, "vllm-omni-private/out/rebuttal/legacy_orig/legacy_sp4_result.json"),
    ("FCFS-SP1", "single-GPU lanes", MAIN, "vllm-omni-gfdit/out/rebuttal/short_orig/sweep_fcfs_sp1_result.json"),
    ("SJF-SP1", "request-size ordering", MAIN, "vllm-omni-gfdit/out/rebuttal/short_orig/sweep_sjf_sp1_result.json"),
    ("SRTF-SP1", "step-level preemption", MAIN, "vllm-omni-gfdit/out/rebuttal/short_orig/sweep_srtf_sp1_result.json"),
    ("EDF-SP1", "deadline ordering", MAIN,
     "vllm-omni-gfdit/out/rebuttal/short_orig_edfsp1/sweep_edf_best_fit_result.json"),
    ("EDF", "dynamic parallelism", MAIN, "vllm-omni-gfdit/out/rebuttal/short_orig/sweep_edf_best_fit_result.json"),
    ("EDF-SP1 @40% load", "", LOAD40, "vllm-omni-gfdit/out/rebuttal/load40_edfsp1/sweep_edf_best_fit_result.json"),
    ("EDF @40% load", "", LOAD40, "vllm-omni-gfdit/out/rebuttal/load40/sweep_edf_best_fit_result.json"),
    ("FCFS-SP4 @40% load", "", LOAD40, "vllm-omni-gfdit/out/rebuttal/load40/sweep_fcfs_sp4_result.json"),
]


def load(bundle: Path, member: str) -> dict:
    with tarfile.open(bundle) as tf:
        return json.load(tf.extractfile(member))


def main() -> None:
    rows = []
    for name, mech, bundle, member in ROWS:
        d = load(bundle, member)
        pc = d["per_class"]
        n = sum(c["n"] for c in pc.values())
        met = sum(c["slo_met"] for c in pc.values())
        rows.append({"configuration": name, "added_mechanism": mech, "slo_pct": round(100 * met / n, 1),
                     "mean_latency_s": round(d["latency_mean"], 1), "completed": d["completed_requests"],
                     "failed": d["failed_requests"], "source": f"{bundle.name}:{member}"})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.with_suffix(".csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    md = ["| Configuration | Added mechanism | SLO (%) | Mean (s) |", "|---|---|---:|---:|"]
    md += [f"| {r['configuration']} | {r['added_mechanism']} | {r['slo_pct']} | {r['mean_latency_s']} |" for r in rows]
    OUT.with_suffix(".md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"Wrote {OUT.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
