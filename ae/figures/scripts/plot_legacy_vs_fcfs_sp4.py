#!/usr/bin/env python3
"""Compare the native legacy path with runtime_v2 pinned to FCFS-SP4."""

from __future__ import annotations

import csv
import io
import json
import tarfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from paper_style import GRID_COLOR, REFERENCE_COLOR, SYSTEM_COLORS


ROOT = Path(__file__).resolve().parents[1]
LEGACY_BUNDLE = ROOT / "logs" / "legacy_baseline_sp4_bundle_20260606.tgz"
WAN_BUNDLE = ROOT / "logs" / "h20_3_workloads_8_policies.tar.gz"
OUT_PREFIX = ROOT / "figs" / "legacy_vs_fcfs_sp4"

SCENARIOS = [
    (
        "WAN short",
        LEGACY_BUNDLE,
        "legacy_baseline_sp4_bundle_20260606/results/wan_short/legacy_sp4_result.json",
        WAN_BUNDLE,
        "./results_nccl/h20bat_short_tl1.20_a2.0_2.5_3.5_seed42/sweep_fcfs_sp4_result.json",
    ),
    (
        "WAN FG",
        LEGACY_BUNDLE,
        "legacy_baseline_sp4_bundle_20260606/results/wan_fg/legacy_sp4_result.json",
        WAN_BUNDLE,
        "./results_nccl/h20bat_foreground_burst_tl1.10_a2.0_2.5_3.5_seed42/sweep_fcfs_sp4_result.json",
    ),
]


def load_member(bundle: Path, member: str) -> dict:
    with tarfile.open(bundle, "r:*") as tf:
        extracted = tf.extractfile(member)
        if extracted is None:
            raise FileNotFoundError(f"{member} in {bundle}")
        with io.TextIOWrapper(extracted, encoding="utf-8") as f:
            return json.load(f)


def slo_attainment(result: dict) -> float:
    total = sum(int(row.get("slo_defined", 0)) for row in result["per_class"].values())
    met = sum(int(row.get("slo_met", 0)) for row in result["per_class"].values())
    return met / total


def load_rows() -> list[dict]:
    rows = []
    for label, legacy_bundle, legacy_member, fcfs_bundle, fcfs_member in SCENARIOS:
        legacy = load_member(legacy_bundle, legacy_member)
        fcfs = load_member(fcfs_bundle, fcfs_member)
        if legacy["failed_requests"] or fcfs["failed_requests"]:
            raise ValueError(f"{label} contains failed requests")
        rows.append(
            {
                "scenario": label,
                "legacy_slo_pct": slo_attainment(legacy) * 100,
                "fcfs_slo_pct": slo_attainment(fcfs) * 100,
                "legacy_mean_s": legacy["latency_mean"],
                "fcfs_mean_s": fcfs["latency_mean"],
                "mean_ratio": fcfs["latency_mean"] / legacy["latency_mean"],
                "legacy_qps": legacy["throughput_qps"],
                "fcfs_qps": fcfs["throughput_qps"],
                "qps_ratio": fcfs["throughput_qps"] / legacy["throughput_qps"],
            }
        )
    return rows


def plot(rows: list[dict]) -> None:
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    plt.rcParams.update(
        {
            "font.size": 7,
            "axes.titlesize": 7,
            "axes.labelsize": 7,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
        }
    )

    x = np.arange(len(rows))
    labels = [row["scenario"] for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(3.35, 1.75))
    specs = [
        ("mean_ratio", "Mean latency / Legacy", "lower is better", SYSTEM_COLORS["latency"]),
        ("qps_ratio", "Throughput / Legacy", "higher is better", SYSTEM_COLORS["throughput"]),
    ]
    for ax, (field, title, direction, color) in zip(axes, specs):
        values = [row[field] for row in rows]
        bars = ax.bar(x, values, width=0.62, color=color, edgecolor="black", linewidth=0.4, hatch="///")
        ax.axhline(1.0, color=REFERENCE_COLOR, linestyle="--", linewidth=0.8, label="Legacy")
        ax.set_title(f"{title}\n({direction})")
        ax.set_ylim(0, max(1.3, max(values) * 1.18))
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=28, ha="right")
        ax.grid(axis="y", color=GRID_COLOR, alpha=0.55, linewidth=0.5)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.035, f"{value:.2f}x", ha="center", va="bottom")
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
    fig.tight_layout(w_pad=0.9)
    fig.savefig(OUT_PREFIX.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(OUT_PREFIX.with_suffix(".png"), dpi=270, bbox_inches="tight")
    plt.close(fig)


def write_csv(rows: list[dict]) -> None:
    with OUT_PREFIX.with_suffix(".csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    rows = load_rows()
    plot(rows)
    write_csv(rows)
    print(f"Wrote {OUT_PREFIX.with_suffix('.pdf')}")
    print(f"Wrote {OUT_PREFIX.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
