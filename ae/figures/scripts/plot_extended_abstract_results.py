#!/usr/bin/env python3
"""Plot a three-metric version of the main-result grid for the extended abstract."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler

from paper_style import GRID_COLOR, POLICY_COLORS


ROW_ORDER = [
    ("H20", "WAN"),
    ("H20", "QWEN"),
    ("A100", "WAN"),
    ("A100", "QWEN"),
]
WORKLOADS = [("short", "Short"), ("fg", "FG burst")]
POLICY_SLOTS = [
    ("edf", "EDF"),
    ("srtf_sp1", "SRTF-SP1"),
    ("srtf_spmax", "SRTF-SPmax"),
    ("fcfs_sp1", "FCFS-SP1"),
    ("legacy", "Legacy"),
]
COLORS = [POLICY_COLORS[slot] for slot, _label in POLICY_SLOTS]
HATCHES = ["", "///", "\\\\\\", "xx", ".."]


def setup_style() -> None:
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    plt.rcParams["axes.prop_cycle"] = cycler("color", COLORS)
    plt.rc("font", size=7)
    plt.rc("axes", titlesize=7)
    plt.rc("axes", labelsize=7)
    plt.rc("xtick", labelsize=6)
    plt.rc("ytick", labelsize=6)
    plt.rc("legend", fontsize=6)


def load_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            parsed: dict[str, object] = dict(row)
            for key in [
                "slo_attainment",
                "latency_mean_s",
                "throughput_qps",
                "mean_latency_norm_to_legacy",
            ]:
                parsed[key] = float(row[key])
            rows.append(parsed)
    return rows


def lookup(rows: list[dict[str, object]]) -> dict[tuple[str, str, str, str], dict[str, object]]:
    return {
        (
            str(row["gpu"]),
            str(row["model"]),
            str(row["workload"]),
            str(row["policy_slot"]),
        ): row
        for row in rows
    }


def plot(rows: list[dict[str, object]], out: Path) -> None:
    data = lookup(rows)
    metric_specs = [
        ("SLO", lambda row: float(row["slo_attainment"]) * 100.0, (0, 105), False),
        ("Mean (s)", lambda row: float(row["latency_mean_s"]), None, True),
        ("QPS", lambda row: float(row["throughput_qps"]), None, False),
    ]

    fig, axes = plt.subplots(len(ROW_ORDER), len(metric_specs), figsize=(5.0, 4.15), sharex=False)
    group_x = np.arange(len(WORKLOADS))
    bar_w = 0.13
    offsets = (np.arange(len(POLICY_SLOTS)) - (len(POLICY_SLOTS) - 1) / 2.0) * bar_w

    for row_idx, (gpu, model) in enumerate(ROW_ORDER):
        for col_idx, (title, value_fn, ylim, log_y) in enumerate(metric_specs):
            ax = axes[row_idx][col_idx]
            all_values = []
            for pidx, (slot, _label) in enumerate(POLICY_SLOTS):
                values = [value_fn(data[(gpu, model, workload, slot)]) for workload, _wlabel in WORKLOADS]
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
                if title == "SLO":
                    for x, value in zip(group_x + offsets[pidx], values):
                        ax.text(x, value + 2.0, f"{value:.0f}", ha="center", va="bottom", fontsize=3.7, rotation=90)

            finite = [value for value in all_values if value > 0]
            if ylim is not None:
                ax.set_ylim(*ylim)
            elif log_y:
                ax.set_yscale("log")
                ax.set_ylim(min(finite) * 0.75, max(finite) * 1.25)
            else:
                ax.set_ylim(0, max(finite) * 1.12)

            if row_idx == 0:
                ax.set_title(title)
            if col_idx == 0:
                ax.set_ylabel(f"{gpu}\n{model}", fontweight="bold")
            ax.grid(axis="y", color=GRID_COLOR, alpha=0.5, linewidth=0.5)
            ax.tick_params(axis="x", length=0)
            ax.set_xticks(group_x)
            if row_idx == len(ROW_ORDER) - 1:
                ax.set_xticklabels([label for _workload, label in WORKLOADS], rotation=25, ha="right")
            else:
                ax.set_xticklabels([])
            for spine in ["top", "right"]:
                ax.spines[spine].set_visible(False)

    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=color, edgecolor="black", linewidth=0.35, hatch=hatch)
        for color, hatch in zip(COLORS, HATCHES)
    ]
    fig.legend(
        handles,
        [label for _slot, label in POLICY_SLOTS],
        frameon=False,
        loc="upper center",
        ncol=5,
        bbox_to_anchor=(0.5, 1.035),
        columnspacing=1.0,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95], w_pad=0.55, h_pad=0.38)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=270, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=Path("figs/main_result_combined.csv"))
    parser.add_argument("--out", type=Path, default=Path("figs/extended_abstract_results.pdf"))
    args = parser.parse_args()
    setup_style()
    plot(load_rows(args.csv), args.out)
    print(f"Wrote {args.out}")
    print(f"Wrote {args.out.with_suffix('.png')}")


if __name__ == "__main__":
    main()
