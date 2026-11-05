#!/usr/bin/env python3
"""Plot a combined H20/A100 main-result metric grid."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from cycler import cycler

import plot_main_result_4gpu as h20_results
import plot_main_result_a100 as a100_results
from paper_style import GRID_COLOR, POLICY_COLORS


WORKLOADS = [("short", "Short"), ("fg", "FG burst")]
ROW_ORDER = [
    ("H20", "WAN"),
    ("H20", "QWEN"),
    ("A100", "WAN"),
    ("A100", "QWEN"),
]
POLICY_SLOTS = [
    ("edf", "EDF"),
    ("srtf_sp1", "SRTF-SP1"),
    ("srtf_spmax", "SRTF-SPmax"),
    ("fcfs_sp1", "FCFS-SP1"),
    ("legacy", "Legacy"),
]
POLICY_MAP = {
    ("H20", "edf"): "edf_best_fit",
    ("H20", "srtf_sp1"): "srtf_sp1",
    ("H20", "srtf_spmax"): "srtf_sp4",
    ("H20", "fcfs_sp1"): "fcfs_sp1",
    ("H20", "legacy"): "legacy",
    ("A100", "edf"): "edf_best_fit",
    ("A100", "srtf_sp1"): "srtf_sp1",
    ("A100", "srtf_spmax"): "srtf_sp8",
    ("A100", "fcfs_sp1"): "fcfs_sp1",
    ("A100", "legacy"): "legacy",
}
COLORS = [POLICY_COLORS[slot] for slot, _label in POLICY_SLOTS]
HATCHES = ["", "///", "\\\\\\", "xx", ".."]


def setup_style() -> None:
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    plt.rcParams["axes.prop_cycle"] = cycler("color", COLORS)
    # PPoPP requires >= 8 pt for all fonts in figures at final size; the grid
    # renders at \textwidth (~7.0 in) so points here are close to final points.
    small = 8.2
    plt.rc("font", size=small)
    plt.rc("axes", titlesize=small)
    plt.rc("axes", labelsize=small)
    plt.rc("xtick", labelsize=small)
    plt.rc("ytick", labelsize=small)
    plt.rc("legend", fontsize=small)
    plt.rc("figure", titlesize=small)


def normalize_rows(h20_rows: list[dict], a100_rows: list[dict]) -> list[dict]:
    rows = []
    for row in h20_rows:
        if row["workload"] not in dict(WORKLOADS):
            continue
        enriched = dict(row)
        enriched["gpu"] = "H20"
        enriched["gpu_count"] = 4
        rows.append(enriched)
    for row in a100_rows:
        if row["workload"] not in dict(WORKLOADS):
            continue
        enriched = dict(row)
        enriched["gpu"] = "A100"
        enriched["gpu_count"] = 8
        rows.append(enriched)
    return rows


def row_lookup(rows: list[dict]) -> dict[tuple[str, str, str, str], dict]:
    return {(r["gpu"], r["model"], r["workload"], r["policy"]): r for r in rows}


def slot_row(lookup: dict, gpu: str, model: str, workload: str, slot: str) -> dict:
    policy = POLICY_MAP[(gpu, slot)]
    return lookup[(gpu, model, workload, policy)]


def write_tables(rows: list[dict], out_prefix: Path) -> None:
    lookup = row_lookup(rows)
    csv_path = out_prefix.with_suffix(".csv")
    fields = [
        "gpu",
        "gpu_count",
        "model",
        "workload",
        "policy_slot",
        "policy_label",
        "source_policy",
        "slo_attainment",
        "latency_mean_s",
        "latency_p50_s",
        "latency_p95_s",
        "latency_p99_s",
        "throughput_qps",
        "completed_requests",
        "failed_requests",
        "mean_latency_norm_to_legacy",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for gpu, model in ROW_ORDER:
            for workload, _wlabel in WORKLOADS:
                legacy = slot_row(lookup, gpu, model, workload, "legacy")
                for slot, label in POLICY_SLOTS:
                    row = slot_row(lookup, gpu, model, workload, slot)
                    writer.writerow(
                        {
                            "gpu": gpu,
                            "gpu_count": row["gpu_count"],
                            "model": model,
                            "workload": workload,
                            "policy_slot": slot,
                            "policy_label": label,
                            "source_policy": row["policy"],
                            "slo_attainment": row["slo_attainment"],
                            "latency_mean_s": row["latency_mean_s"],
                            "latency_p50_s": row["latency_p50_s"],
                            "latency_p95_s": row["latency_p95_s"],
                            "latency_p99_s": row["latency_p99_s"],
                            "throughput_qps": row["throughput_qps"],
                            "completed_requests": row["completed_requests"],
                            "failed_requests": row["failed_requests"],
                            "mean_latency_norm_to_legacy": row["latency_mean_s"] / legacy["latency_mean_s"],
                        }
                    )

    md_path = out_prefix.with_suffix(".md")
    with md_path.open("w", encoding="utf-8") as f:
        f.write("# Combined H20/A100 Main Result\n\n")
        f.write("| gpu | model | workload | policy | source | SLO % | mean s | p50 s | p95 s | p99 s | qps | fail |\n")
        f.write("| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n")
        for gpu, model in ROW_ORDER:
            for workload, _wlabel in WORKLOADS:
                for slot, label in POLICY_SLOTS:
                    row = slot_row(lookup, gpu, model, workload, slot)
                    f.write(
                        f"| {gpu} | {model} | {workload} | {label} | {row['policy']} | "
                        f"{row['slo_attainment'] * 100:.1f} | {row['latency_mean_s']:.1f} | "
                        f"{row['latency_p50_s']:.1f} | {row['latency_p95_s']:.1f} | "
                        f"{row['latency_p99_s']:.1f} | {row['throughput_qps']:.3f} | "
                        f"{row['failed_requests']} |\n"
                    )


def plot_metric_grid(rows: list[dict], out_prefix: Path) -> None:
    lookup = row_lookup(rows)
    metric_specs = [
        ("SLO (%)", lambda row: row["slo_attainment"] * 100.0, (0, 105), False),
        ("Mean (s)", lambda row: row["latency_mean_s"], None, True),
        ("Median (s)", lambda row: row["latency_p50_s"], None, True),
        ("P95 (s)", lambda row: row["latency_p95_s"], None, True),
        ("QPS", lambda row: row["throughput_qps"], None, False),
    ]
    fig, axes = plt.subplots(len(ROW_ORDER), len(metric_specs), figsize=(7.15, 2.9), sharex=False)
    workloads = [workload for workload, _label in WORKLOADS]
    workload_labels = [label for _workload, label in WORKLOADS]
    group_x = np.arange(len(workloads))
    bar_w = 0.13
    offsets = (np.arange(len(POLICY_SLOTS)) - (len(POLICY_SLOTS) - 1) / 2.0) * bar_w

    for row_idx, (gpu, model) in enumerate(ROW_ORDER):
        for col_idx, (title, value_fn, ylim, log_y) in enumerate(metric_specs):
            ax = axes[row_idx][col_idx]
            all_values = []
            for pidx, (slot, _label) in enumerate(POLICY_SLOTS):
                values = [value_fn(slot_row(lookup, gpu, model, workload, slot)) for workload in workloads]
                all_values.extend(values)
                ax.bar(
                    group_x + offsets[pidx],
                    values,
                    width=bar_w * 0.92,
                    color=COLORS[pidx],
                    edgecolor="black",
                    linewidth=0.22,
                    hatch=HATCHES[pidx],
                )
                # No per-bar value labels: at this panel density they cannot
                # meet the 8 pt floor; exact numbers live in the prose and in
                # figs/main_result_combined.md.
            finite_values = [value for value in all_values if math.isfinite(value)]
            if ylim is not None:
                ax.set_ylim(*ylim)
            elif log_y:
                ymin = max(min(v for v in finite_values if v > 0) * 0.75, 1e-3)
                ax.set_ylim(ymin, max(finite_values) * 1.25)
            else:
                ax.set_ylim(0, max(finite_values) * 1.12)

            if log_y:
                ax.set_yscale("log")
                # Decade ticks only; unlabeled minors kill the "6 x 10^2"
                # clutter of the previous version.
                ax.yaxis.set_major_locator(mticker.LogLocator(base=10.0, numticks=5))
                ax.yaxis.set_minor_locator(mticker.LogLocator(base=10.0, subs=(2.0, 5.0), numticks=10))
                ax.yaxis.set_minor_formatter(mticker.NullFormatter())
                ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
                ax.yaxis.get_major_formatter().set_scientific(False)
            if row_idx == 0:
                ax.set_title(title)
            if col_idx == 0:
                ax.set_ylabel(f"{gpu}\n{model}", fontweight="bold")
            ax.grid(axis="y", color=GRID_COLOR, alpha=0.5, linewidth=0.5)
            ax.tick_params(axis="x", length=0)
            ax.set_xticks(group_x)
            if row_idx == len(ROW_ORDER) - 1:
                ax.set_xticklabels(workload_labels, rotation=25, ha="right")
            else:
                ax.set_xticklabels([])
            for spine in ["top", "right"]:
                ax.spines[spine].set_visible(False)

    policy_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=color, edgecolor="black", linewidth=0.35, hatch=hatch)
        for color, hatch in zip(COLORS, HATCHES)
    ]
    fig.legend(
        policy_handles,
        [label for _slot, label in POLICY_SLOTS],
        frameon=False,
        loc="upper center",
        ncol=5,
        bbox_to_anchor=(0.5, 1.035),
        columnspacing=1.1,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95], w_pad=0.55, h_pad=0.38)
    fig.savefig(out_prefix.with_name(out_prefix.name + "_metric_grid.pdf"), bbox_inches="tight")
    fig.savefig(out_prefix.with_name(out_prefix.name + "_metric_grid.png"), dpi=270, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--a100-bundle", type=Path, default=Path("logs/a100_full_benchmark.tgz"))
    parser.add_argument("--out-prefix", type=Path, default=Path("figs/main_result_combined"))
    args = parser.parse_args()

    setup_style()
    root = args.root.resolve()
    out_prefix = args.out_prefix if args.out_prefix.is_absolute() else root / args.out_prefix
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    rows = normalize_rows(
        h20_results.load_rows(root),
        a100_results.load_rows(args.a100_bundle),
    )
    write_tables(rows, out_prefix)
    plot_metric_grid(rows, out_prefix)
    print(f"Wrote {out_prefix.with_name(out_prefix.name + '_metric_grid.pdf')}")
    print(f"Wrote {out_prefix.with_name(out_prefix.name + '_metric_grid.png')}")
    print(f"Wrote {out_prefix.with_suffix('.csv')}")
    print(f"Wrote {out_prefix.with_suffix('.md')}")


if __name__ == "__main__":
    main()
