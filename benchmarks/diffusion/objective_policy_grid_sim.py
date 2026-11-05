# SPDX-License-Identifier: Apache-2.0
"""Objective-dependent policy grid (simulator).

Reproduces the "objective-dependent policy" figure via the runtime_v2 policy
simulator, using the measured H20 4-card workloads + cost models. EDF (the
adaptive policy) is intentionally excluded; non-preemptive SJF is added.

For every (workload x policy x SP-width) cell it runs the discrete-event
simulator and reports, over ALL requests, the metrics the figure ranks on:
SLO attainment, mean / median / p95 / p99 latency, and throughput. Per-class
(S/M/L) numbers are kept too so the plot can re-pick the scope later.

Policies compared (EDF removed, SJF added): {FCFS, SRTF, SJF} x {SP1, SP4}.
Note: FCFS-SP4 is the "Legacy" static-SP4-FIFO baseline.

Because none of these are adaptive, there is no cross-group migration / reshard
in any run -- so the simulator tracks real H20 closely for exactly this set
(the known sim<->real gap is driven by EDF migration orchestration).

Usage:
  PYTHONPATH=. .venv/bin/python benchmarks/diffusion/objective_policy_grid_sim.py \
      --out-dir out/objective_sim
  # add --include-qwen to also run the QWEN rows.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from benchmarks.diffusion.runtime_v2_mixed_priority_runner import (
    _build_topology,
    _percentile,
    _run_one_policy,
)
from benchmarks.diffusion.runtime_v2_policy_simulator import (
    CostModelStore,
    _request_from_mapping,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

# --- grid definition -------------------------------------------------------

# Measured H20 4-card workloads (the "实测配置").
WAN_WORKLOADS = [
    {
        "row": "WAN short",
        "workload": "out/stress/h20bat_short_tl1.20_a2.0_2.5_3.5_seed42.json",
        "cost_model": "cost-model/wan22-ti2v-5b-h20",
    },
    {
        "row": "WAN FG burst",
        "workload": "out/stress/h20bat_foreground_burst_tl1.10_a2.0_2.5_3.5_seed42.json",
        "cost_model": "cost-model/wan22-ti2v-5b-h20",
    },
]

QWEN_WORKLOADS = [
    {
        "row": "QWEN short",
        "workload": "out/stress/qwenbat_short_tl1.20_seed42.json",
        "cost_model": "cost-model/qwen-image",
    },
    {
        # canonical QWEN battery load (commit 2226224): fg_burst calibrated at
        # TL=0.90, not 1.10 -- the load the real experiment actually replayed.
        "row": "QWEN FG burst",
        "workload": "out/stress/qwenbat_foreground_burst_tl0.90_seed42.json",
        "cost_model": "cost-model/qwen-image",
    },
]

WORLD_SIZE = 4

# SP-width -> topology groups (tp=1 so the cost-model key matches tp1_ulyssesK).
SP_VARIANTS = {
    "SP1": [
        {"group_id": f"g{r}", "ranks": [r], "tp": 1, "ulysses_degree": 1, "ring_degree": 1, "cfg": 1}
        for r in range(WORLD_SIZE)
    ],
    "SP4": [
        {"group_id": "g0", "ranks": list(range(WORLD_SIZE)), "tp": 1,
         "ulysses_degree": WORLD_SIZE, "ring_degree": 1, "cfg": 1},
    ],
}

# base policy -> simulator factory spec
BASE_POLICIES = {
    "FCFS": "benchmarks.diffusion.policies:make_fcfs",
    "SRTF": "benchmarks.diffusion.policies:make_srtf",
    "SJF": "benchmarks.diffusion.policies:make_sjf",
}

# Overall (all-request) metrics. direction: +1 = higher is better, -1 = lower.
METRICS = [
    ("SLO", "slo_pct", +1, "{:.1f}%"),
    ("Mean", "mean_s", -1, "{:.0f}s"),
    ("Median", "median_s", -1, "{:.0f}s"),
    ("P95", "p95_s", -1, "{:.0f}s"),
    ("P99", "p99_s", -1, "{:.0f}s"),
    ("Throughput", "throughput_rpm", +1, "{:.2f}/min"),
]

# Per-priority-class objectives -- the figure's columns. Each is (display, class
# or None for overall, metric_key, direction, fmt). This is the framing that
# actually differentiates the static policies: foreground (S) wants preemptive
# short-first, background (L) wants no-starvation FIFO, the tail wants no
# reordering -- so different objectives genuinely pick different policies.
PER_CLASS_OBJECTIVES = [
    ("前台SLO(S)", "S", "slo_pct", +1, "{:.0f}%"),
    ("前台延迟(S median)", "S", "median_s", -1, "{:.0f}s"),
    ("中台延迟(M median)", "M", "median_s", -1, "{:.0f}s"),
    ("后台延迟(L median)", "L", "median_s", -1, "{:.0f}s"),
    ("整体尾(overall P99)", None, "p99_s", -1, "{:.0f}s"),
]

# Full menu for column-picking: every scope x every metric.
MENU_SCOPES = [("S", "S(前台)"), ("M", "M(中台)"), ("L", "L(后台)"), (None, "全局overall")]
MENU_METRICS = [
    ("SLO", "slo_pct", +1, "{:.0f}%"),
    ("mean", "mean_s", -1, "{:.0f}s"),
    ("median", "median_s", -1, "{:.0f}s"),
    ("p95", "p95_s", -1, "{:.0f}s"),
    ("p99", "p99_s", -1, "{:.0f}s"),
]

# args consumed by _run_one_policy (no migration -> reshard values are inert).
SIM_ARGS = SimpleNamespace(
    reshard_ms=3.0,
    reshard_samples_json=None,
    launch_end_before_exec_end_ms=0.05,
    exec_start_delay_ms=0.005,
    trace_out_dir=None,
)


def _metrics_from_result(result: dict[str, Any]) -> dict[str, Any]:
    """Overall (all-request) + per-class metrics the figure ranks on."""
    per_req = result["per_request"]
    overall = result["overall"]
    total = overall["total_requests"]
    makespan_s = overall["makespan_ms"] / 1000.0

    def _agg(items: list[dict[str, Any]], denom: int) -> dict[str, float]:
        lat = [it["latency_ms"] / 1000.0 for it in items]
        met = sum(1 for it in items if it["deadline_met"])
        return {
            "n": len(items),
            "slo_pct": 100.0 * met / denom if denom else 0.0,
            "mean_s": statistics.mean(lat) if lat else 0.0,
            "median_s": statistics.median(lat) if lat else 0.0,
            "p95_s": _percentile(lat, 95),
            "p99_s": _percentile(lat, 99),
        }

    out = _agg(per_req, total)
    out["completed"] = overall["completed_requests"]
    out["total"] = total
    out["makespan_s"] = makespan_s
    out["throughput_rpm"] = (overall["completed_requests"] / makespan_s * 60.0) if makespan_s else 0.0

    by_class: dict[str, dict[str, float]] = {}
    classes = {it["request_class"] for it in per_req}
    for klass in sorted(classes):
        items = [it for it in per_req if it["request_class"] == klass]
        by_class[klass] = _agg(items, len(items))
    out["by_class"] = by_class
    return out


def _run_cell(workload_path: Path, cost_model_dir: Path, sp_label: str,
              base_label: str, policy_spec: str) -> dict[str, Any]:
    topology = _build_topology(
        json.dumps(SP_VARIANTS[sp_label]), None, "sliding", world_size=WORLD_SIZE
    )
    cost_models = CostModelStore.load_dir(str(cost_model_dir))
    workload_data = json.loads(workload_path.read_text())
    requests = [_request_from_mapping(item) for item in workload_data.get("requests", [])]
    name = f"{base_label}-{sp_label}"
    result = _run_one_policy(policy_spec, name, requests, topology, cost_models, SIM_ARGS)
    metrics = _metrics_from_result(result)
    metrics["policy"] = name
    metrics["base"] = base_label
    metrics["sp"] = sp_label
    return metrics


def _cell_value(cell: dict[str, Any], klass: str | None, key: str) -> float:
    """Metric value for a policy cell, overall (klass=None) or per-class."""
    if klass is None:
        return cell[key]
    return cell["by_class"][klass][key]


def _best_cell(cells: list[dict[str, Any]], klass: str | None, key: str,
               direction: int) -> dict[str, Any]:
    cand = [c for c in cells if klass is None or klass in c["by_class"]]
    return (max if direction > 0 else min)(
        cand, key=lambda c: _cell_value(c, klass, key)
    )


def _objective_menu(rows: list[dict[str, Any]]) -> str:
    """Menu for column-picking: one row per (scope x metric) candidate objective,
    one column per workload, cell = winning policy + value (+ spread across the
    6 policies so you can see which objectives actually differentiate)."""
    header = "| 目标(scope·metric) | " + " | ".join(r["row"] for r in rows) + " |"
    sep = "|" + "|".join(["---"] * (len(rows) + 1)) + "|"
    lines = [header, sep]
    for scope_key, scope_label in MENU_SCOPES:
        for mlabel, key, direction, fmt in MENU_METRICS:
            name = f"{scope_label} {mlabel}"
            cells = [name]
            for r in rows:
                cand = [c for c in r["cells"] if scope_key is None or scope_key in c["by_class"]]
                vals = [_cell_value(c, scope_key, key) for c in cand]
                best = _best_cell(cand, scope_key, key, direction)
                bv = _cell_value(best, scope_key, key)
                spread = max(vals) - min(vals) if vals else 0.0
                cells.append(f"{best['policy']} {fmt.format(bv)} (Δ{fmt.format(spread)})")
            lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _winner_table(rows: list[dict[str, Any]], columns) -> str:
    """Markdown table: row=workload, col=objective, cell='<winning policy> <value>'.

    `columns` is either the overall METRICS shape (label, key, dir, fmt) or the
    per-class shape (label, class, key, dir, fmt); both are normalized here.
    """
    norm = [(c[0], None, c[1], c[2], c[3]) if len(c) == 4 else c for c in columns]
    header = "| Workload | " + " | ".join(c[0] for c in norm) + " |"
    sep = "|" + "|".join(["---"] * (len(norm) + 1)) + "|"
    lines = [header, sep]
    for r in rows:
        cells = [r["row"]]
        for _label, klass, key, direction, fmt in norm:
            best = _best_cell(r["cells"], klass, key, direction)
            cells.append(f"**{best['policy']}** {fmt.format(_cell_value(best, klass, key))}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default="out/objective_sim")
    ap.add_argument("--include-qwen", action="store_true",
                    help="also run the QWEN rows (off by default)")
    args = ap.parse_args()

    workloads = list(WAN_WORKLOADS)
    if args.include_qwen:
        workloads += QWEN_WORKLOADS

    out_dir = REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    grid: list[dict[str, Any]] = []
    table_rows: list[dict[str, Any]] = []
    for wl in workloads:
        wl_path = REPO_ROOT / wl["workload"]
        cm_dir = REPO_ROOT / wl["cost_model"]
        cells: list[dict[str, Any]] = []
        for sp_label in SP_VARIANTS:
            for base_label, spec in BASE_POLICIES.items():
                print(f"[grid] {wl['row']:>14} | {base_label}-{sp_label} ...", flush=True)
                m = _run_cell(wl_path, cm_dir, sp_label, base_label, spec)
                m["row"] = wl["row"]
                m["workload"] = wl["workload"]
                cells.append(m)
                grid.append(m)
        table_rows.append({"row": wl["row"], "cells": cells})

    # full detail
    (out_dir / "grid_results.json").write_text(json.dumps(grid, indent=2))

    # flat CSV (overall metrics per cell)
    with (out_dir / "metrics_long.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["workload", "policy", "base", "sp", "n", "completed", "total",
                    "slo_pct", "mean_s", "median_s", "p95_s", "p99_s",
                    "throughput_rpm", "makespan_s"])
        for m in grid:
            w.writerow([m["row"], m["policy"], m["base"], m["sp"], m["n"],
                        m["completed"], m["total"],
                        f"{m['slo_pct']:.3f}", f"{m['mean_s']:.3f}",
                        f"{m['median_s']:.3f}", f"{m['p95_s']:.3f}", f"{m['p99_s']:.3f}",
                        f"{m['throughput_rpm']:.4f}", f"{m['makespan_s']:.3f}"])

    # full metrics at every scope (S/M/L/overall) x metric -- the column menu.
    with (out_dir / "all_metrics_long.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["workload", "policy", "base", "sp", "scope", "n",
                    "slo_pct", "mean_s", "median_s", "p95_s", "p99_s"])
        for m in grid:
            scopes = [("overall", m)] + [(k, m["by_class"][k]) for k in sorted(m["by_class"])]
            for scope, d in scopes:
                w.writerow([m["row"], m["policy"], m["base"], m["sp"], scope,
                            d.get("n", m["total"]),
                            f"{d['slo_pct']:.3f}", f"{d['mean_s']:.3f}",
                            f"{d['median_s']:.3f}", f"{d['p95_s']:.3f}", f"{d['p99_s']:.3f}"])

    # plot-ready winners CSV (per-class objective framing -- the figure's data)
    with (out_dir / "figure_winners.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["workload", "objective", "scope", "metric", "winner_policy",
                    "value", "value_str"])
        for tr in table_rows:
            for label, klass, key, direction, fmt in PER_CLASS_OBJECTIVES:
                best = _best_cell(tr["cells"], klass, key, direction)
                v = _cell_value(best, klass, key)
                w.writerow([tr["row"], label, klass or "overall", key,
                            best["policy"], f"{v:.3f}", fmt.format(v)])

    # winner table (markdown) -- the figure's data
    md = ["# Objective-dependent policy (simulator, H20 4-card config)",
          "",
          "EDF excluded; non-preemptive SJF added. FCFS-SP4 == the \"Legacy\" "
          "static-SP4-FIFO baseline. No migration in any run (static SP) so "
          "sim tracks real H20 for this policy set.",
          "",
          "## Figure data: winner per (workload x per-class objective)",
          "",
          _winner_table(table_rows, PER_CLASS_OBJECTIVES),
          "",
          "## Reference: winner per (workload x overall objective)",
          "",
          _winner_table(table_rows, METRICS),
          "",
          "## 指标菜单: 每个 scope×metric 候选目标的获胜者 (Δ=6 policy 间极差, 越大越能区分)",
          "",
          _objective_menu(table_rows),
          "",
          "## All policies (per workload)",
          ""]
    for tr in table_rows:
        md.append(f"### {tr['row']}")
        md.append("")
        md.append("| Policy | SLO | Mean | Median | P95 | P99 | Throughput | n(comp/total) |")
        md.append("|---|---|---|---|---|---|---|---|")
        for c in tr["cells"]:
            md.append(
                f"| {c['policy']} | {c['slo_pct']:.1f}% | {c['mean_s']:.0f}s | "
                f"{c['median_s']:.0f}s | {c['p95_s']:.0f}s | {c['p99_s']:.0f}s | "
                f"{c['throughput_rpm']:.2f}/min | {c['completed']}/{c['total']} |"
            )
        md.append("")
    (out_dir / "winners.md").write_text("\n".join(md))

    print("\n=== 指标菜单: 候选目标(scope×metric)→ 各 workload 获胜者 (Δ=policy间极差) ===")
    print(_objective_menu(table_rows))
    print(f"\nwrote: {out_dir/'grid_results.json'}")
    print(f"wrote: {out_dir/'all_metrics_long.csv'}  <- 全 scope×metric 原始值")
    print(f"wrote: {out_dir/'metrics_long.csv'}")
    print(f"wrote: {out_dir/'figure_winners.csv'}")
    print(f"wrote: {out_dir/'winners.md'}")


if __name__ == "__main__":
    main()
