#!/usr/bin/env python3
"""Plot A100 8-GPU main-result metric grid from the full benchmark bundle."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import tarfile
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler

from paper_style import GRID_COLOR, POLICY_COLORS


POLICIES = [
    ("edf_best_fit", "EDF"),
    ("srtf_sp1", "SRTF-SP1"),
    ("srtf_sp8", "SRTF-SP8"),
    ("fcfs_sp1", "FCFS-SP1"),
    ("legacy", "Legacy"),
]
MODELS = ["WAN", "QWEN"]
WORKLOADS = [("short", "Short"), ("fg", "FG burst")]
COLORS = [POLICY_COLORS[policy] for policy, _label in POLICIES]
HATCHES = ["", "///", "\\\\\\", "xx", ".."]


@dataclass(frozen=True)
class Source:
    model: str
    workload: str
    policy: str
    bundle: Path
    member: str


def setup_style() -> None:
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    plt.rcParams["axes.prop_cycle"] = cycler("color", COLORS)
    small = 8
    plt.rc("font", size=small)
    plt.rc("axes", titlesize=small)
    plt.rc("axes", labelsize=small)
    plt.rc("xtick", labelsize=small)
    plt.rc("ytick", labelsize=small)
    plt.rc("legend", fontsize=small)
    plt.rc("figure", titlesize=small)


def a100_sources(bundle: Path) -> list[Source]:
    paths = {
        ("WAN", "short", "edf_best_fit"): "wan_a100_bundle/out/battery8a100_wan_short/wanbat8a100_short_tl1.05_seed42/sweep_edf_best_fit_result.json",
        ("WAN", "short", "srtf_sp1"): "wan_a100_bundle/out/battery8a100_wan_baselines/wanbat8a100_short_tl1.05_seed42/sweep_srtf_sp1_result.json",
        ("WAN", "short", "srtf_sp8"): "wan_a100_bundle/out/battery8a100_wan_baselines/wanbat8a100_short_tl1.05_seed42/sweep_srtf_result.json",
        ("WAN", "short", "fcfs_sp1"): "wan_a100_bundle/out/battery8a100_wan_baselines/wanbat8a100_short_tl1.05_seed42/sweep_fcfs_sp1_result.json",
        ("WAN", "short", "legacy"): "a100_session2/out/battery8a100_wan_legacy/short/legacy_sp8_result.json",
        ("WAN", "fg", "edf_best_fit"): "wan_a100_bundle/out/battery8a100_wan_rest/wanbat8a100_foreground_burst_tl0.9_seed42/sweep_edf_best_fit_result.json",
        ("WAN", "fg", "srtf_sp1"): "wan_a100_bundle/out/battery8a100_wan_baselines/wanbat8a100_foreground_burst_tl0.9_seed42/sweep_srtf_sp1_result.json",
        ("WAN", "fg", "srtf_sp8"): "wan_a100_bundle/out/battery8a100_wan_baselines/wanbat8a100_foreground_burst_tl0.9_seed42/sweep_srtf_result.json",
        ("WAN", "fg", "fcfs_sp1"): "wan_a100_bundle/out/battery8a100_wan_baselines/wanbat8a100_foreground_burst_tl0.9_seed42/sweep_fcfs_sp1_result.json",
        ("WAN", "fg", "legacy"): "a100_session2/out/battery8a100_wan_legacy/fg/legacy_sp8_result.json",
        ("QWEN", "short", "edf_best_fit"): "a100_session2/out/battery8a100_qwen_origalpha/qwenbat8a100_short_tl1.0_origalpha_seed42/sweep_edf_best_fit_result.json",
        ("QWEN", "short", "srtf_sp1"): "a100_session2/out/battery8a100_qwen_origalpha/qwenbat8a100_short_tl1.0_origalpha_seed42/sweep_srtf_sp1_result.json",
        ("QWEN", "short", "srtf_sp8"): "a100_session2/out/battery8a100_qwen_origalpha/qwenbat8a100_short_tl1.0_origalpha_seed42/sweep_srtf_result.json",
        ("QWEN", "short", "fcfs_sp1"): "a100_session2/out/battery8a100_qwen_origalpha/qwenbat8a100_short_tl1.0_origalpha_seed42/sweep_fcfs_sp1_result.json",
        ("QWEN", "short", "legacy"): "a100_session2/out/battery8a100_qwen_origalpha/qwenbat8a100_short_tl1.0_origalpha_seed42/legacy_sp8_result.json",
        ("QWEN", "fg", "edf_best_fit"): "a100_session2/out/battery8a100_qwen_origalpha/qwenbat8a100_fg_tl0.7_origalpha_seed42/sweep_edf_best_fit_result.json",
        ("QWEN", "fg", "srtf_sp1"): "a100_session2/out/battery8a100_qwen_origalpha/qwenbat8a100_fg_tl0.7_origalpha_seed42/sweep_srtf_sp1_result.json",
        ("QWEN", "fg", "srtf_sp8"): "a100_session2/out/battery8a100_qwen_origalpha/qwenbat8a100_fg_tl0.7_origalpha_seed42/sweep_srtf_result.json",
        ("QWEN", "fg", "fcfs_sp1"): "a100_session2/out/battery8a100_qwen_origalpha/qwenbat8a100_fg_tl0.7_origalpha_seed42/sweep_fcfs_sp1_result.json",
        ("QWEN", "fg", "legacy"): "a100_session2/out/battery8a100_qwen_origalpha/qwenbat8a100_fg_tl0.7_origalpha_seed42/legacy_sp8_result.json",
    }
    return [
        Source(model=model, workload=workload, policy=policy, bundle=bundle, member=member)
        for (model, workload, policy), member in paths.items()
    ]


def read_json(source: Source) -> dict:
    with tarfile.open(source.bundle, "r:*") as tf:
        extracted = tf.extractfile(source.member)
        if extracted is None:
            raise FileNotFoundError(source.member)
        with io.TextIOWrapper(extracted, encoding="utf-8") as f:
            return json.load(f)


def summarize_result(source: Source, data: dict) -> dict:
    total = 0
    met = 0
    for cls in data.get("per_class", {}).values():
        total += int(cls.get("slo_defined", 0))
        met += int(cls.get("slo_met", 0))
    slo = met / total if total else math.nan
    return {
        "gpu": "A100",
        "gpu_count": 8,
        "model": source.model,
        "workload": source.workload,
        "policy": source.policy,
        "slo_attainment": slo,
        "latency_mean_s": float(data.get("latency_mean", math.nan)),
        "latency_p50_s": float(data.get("latency_p50", data.get("latency_median", math.nan))),
        "latency_p95_s": float(data.get("latency_p95", math.nan)),
        "latency_p99_s": float(data.get("latency_p99", math.nan)),
        "throughput_qps": float(data.get("throughput_qps", math.nan)),
        "completed_requests": int(data.get("completed_requests", 0)),
        "failed_requests": int(data.get("failed_requests", 0)),
    }


def load_rows(bundle: Path) -> list[dict]:
    return [summarize_result(source, read_json(source)) for source in a100_sources(bundle)]


def row_lookup(rows: list[dict]) -> dict[tuple[str, str, str], dict]:
    return {(r["model"], r["workload"], r["policy"]): r for r in rows}


def write_tables(rows: list[dict], out_prefix: Path) -> None:
    lookup = row_lookup(rows)
    csv_path = out_prefix.with_suffix(".csv")
    fields = list(rows[0].keys()) + ["mean_latency_norm_to_legacy"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            enriched = dict(row)
            legacy = lookup[(row["model"], row["workload"], "legacy")]
            enriched["mean_latency_norm_to_legacy"] = row["latency_mean_s"] / legacy["latency_mean_s"]
            writer.writerow(enriched)

    md_path = out_prefix.with_suffix(".md")
    with md_path.open("w", encoding="utf-8") as f:
        f.write("# A100 8-GPU Main Result Draft\n\n")
        f.write("| model | workload | policy | SLO % | mean s | p50 s | p95 s | p99 s | qps | fail | mean/legacy |\n")
        f.write("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n")
        for model in MODELS:
            for workload, _wlabel in WORKLOADS:
                legacy = lookup[(model, workload, "legacy")]
                for policy, label in POLICIES:
                    row = lookup[(model, workload, policy)]
                    f.write(
                        f"| {model} | {workload} | {label} | "
                        f"{row['slo_attainment'] * 100:.1f} | "
                        f"{row['latency_mean_s']:.1f} | {row['latency_p50_s']:.1f} | "
                        f"{row['latency_p95_s']:.1f} | {row['latency_p99_s']:.1f} | "
                        f"{row['throughput_qps']:.3f} | {row['failed_requests']} | "
                        f"{row['latency_mean_s'] / legacy['latency_mean_s']:.2f} |\n"
                    )


def plot_metric_grid(rows: list[dict], out_prefix: Path) -> None:
    lookup = row_lookup(rows)
    metric_specs = [
        ("slo_attainment", "SLO", lambda row: row["slo_attainment"] * 100.0, (0, 105), False, "%"),
        ("latency_mean_s", "Mean", lambda row: row["latency_mean_s"], None, True, "s"),
        ("latency_p50_s", "Median", lambda row: row["latency_p50_s"], None, True, "s"),
        ("latency_p95_s", "P95", lambda row: row["latency_p95_s"], None, True, "s"),
        ("latency_p99_s", "P99", lambda row: row["latency_p99_s"], None, True, "s"),
        ("throughput_qps", "QPS", lambda row: row["throughput_qps"], None, False, ""),
    ]

    fig, axes = plt.subplots(2, 6, figsize=(7.15, 3.05), sharex=False)
    bar_w = 0.13
    workloads = [workload for workload, _label in WORKLOADS]
    workload_labels = [label for _workload, label in WORKLOADS]
    group_x = np.arange(len(workloads))
    offsets = (np.arange(len(POLICIES)) - (len(POLICIES) - 1) / 2.0) * bar_w

    for row_idx, model in enumerate(MODELS):
        for col_idx, (_metric, title, value_fn, ylim, log_y, unit) in enumerate(metric_specs):
            ax = axes[row_idx][col_idx]
            for pidx, (policy, _label) in enumerate(POLICIES):
                values = [value_fn(lookup[(model, workload, policy)]) for workload in workloads]
                ax.bar(
                    group_x + offsets[pidx],
                    values,
                    width=bar_w * 0.92,
                    color=COLORS[pidx],
                    edgecolor="black",
                    linewidth=0.25,
                    hatch=HATCHES[pidx],
                )
                if title == "SLO":
                    for x, value in zip(group_x + offsets[pidx], values):
                        ax.text(x, value + 2.0, f"{value:.0f}", ha="center", va="bottom", fontsize=4.6, rotation=90)
            if row_idx == 0:
                ax.set_title(f"{title} ({unit})" if unit else title)
            ax.grid(axis="y", color=GRID_COLOR, alpha=0.5, linewidth=0.55)
            if log_y:
                ax.set_yscale("log")
            if ylim is not None:
                ax.set_ylim(*ylim)
            else:
                all_values = [
                    value_fn(lookup[(model, workload, policy)])
                    for workload in workloads
                    for policy, _label in POLICIES
                ]
                if log_y:
                    ymin = max(min(v for v in all_values if v > 0) * 0.75, 1e-3)
                    ax.set_ylim(ymin, max(all_values) * 1.25)
                else:
                    ax.set_ylim(0, max(all_values) * 1.12)
            ax.tick_params(axis="x", length=0)
            ax.set_xticks(group_x)
            if row_idx == len(MODELS) - 1:
                ax.set_xticklabels(workload_labels, rotation=25, ha="right")
            else:
                ax.set_xticklabels([])
            if col_idx == 0:
                ax.set_ylabel(model, fontweight="bold")
            for spine in ["top", "right"]:
                ax.spines[spine].set_visible(False)

    handles = []
    labels = []
    for color, hatch, (_policy, label) in zip(COLORS, HATCHES, POLICIES):
        patch = plt.Rectangle((0, 0), 1, 1, facecolor=color, edgecolor="black", linewidth=0.35, hatch=hatch)
        handles.append(patch)
        labels.append(label)
    fig.legend(handles, labels, frameon=False, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout(rect=[0, 0, 1, 0.92], w_pad=0.6, h_pad=0.65)
    fig.savefig(out_prefix.with_name(out_prefix.name + "_metric_grid.pdf"), bbox_inches="tight")
    fig.savefig(out_prefix.with_name(out_prefix.name + "_metric_grid.png"), dpi=260, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, default=Path("logs/a100_full_benchmark.tgz"))
    parser.add_argument("--out-prefix", type=Path, default=Path("figs/main_result_a100"))
    args = parser.parse_args()

    setup_style()
    out_prefix = args.out_prefix.resolve()
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.bundle)
    write_tables(rows, out_prefix)
    plot_metric_grid(rows, out_prefix)
    print(f"Wrote {out_prefix.with_name(out_prefix.name + '_metric_grid.pdf')}")
    print(f"Wrote {out_prefix.with_name(out_prefix.name + '_metric_grid.png')}")
    print(f"Wrote {out_prefix.with_suffix('.csv')}")
    print(f"Wrote {out_prefix.with_suffix('.md')}")


if __name__ == "__main__":
    main()
