#!/usr/bin/env python3
"""Plot a 4-GPU main-result draft from existing WAN and QWEN logs."""

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
import matplotlib.patheffects as path_effects
import numpy as np
from cycler import cycler

from paper_style import GRID_COLOR, POLICY_COLORS, REFERENCE_COLOR, SLO_CMAP


POLICIES = [
    ("edf_best_fit", "EDF"),
    ("srtf_sp1", "SRTF-SP1"),
    ("srtf_sp4", "SRTF-SP4"),
    ("fcfs_sp1", "FCFS-SP1"),
    ("legacy", "Legacy"),
]
WORKLOADS = [
    ("short", "Short"),
    ("inflight", "Inflight"),
    ("fg", "FG burst"),
]
DEFAULT_PLOT_WORKLOADS = ["short", "fg"]
MODELS = ["WAN", "QWEN"]

COLORS = [POLICY_COLORS[policy] for policy, _label in POLICIES]
HATCHES = ["", "///", "\\\\\\", "xx", ".."]


@dataclass(frozen=True)
class Source:
    model: str
    workload: str
    policy: str
    path: Path
    member: str | None = None


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


def read_json(source: Source) -> dict:
    if source.member is None:
        return json.loads(source.path.read_text(encoding="utf-8"))
    with tarfile.open(source.path, "r:*") as tf:
        member = tf.extractfile(source.member)
        if member is None:
            raise FileNotFoundError(source.member)
        with io.TextIOWrapper(member, encoding="utf-8") as f:
            return json.load(f)


def wan_profile_dir(workload: str) -> str:
    return {
        "short": "h20bat_short_tl1.20_a2.0_2.5_3.5_seed42",
        "inflight": "h20bat_inflight_burst_tl0.90_a2.0_2.5_3.5_seed42",
        "fg": "h20bat_foreground_burst_tl1.10_a2.0_2.5_3.5_seed42",
    }[workload]


def qwen_profile_dir(workload: str) -> str:
    return {
        "short": "short_tl1.20",
        "inflight": "inflight_burst_tl0.70",
        "fg": "fg_burst_tl0.90",
    }[workload]


def legacy_profile_dir(model: str, workload: str) -> str:
    return f"{model.lower()}_{workload}"


def discover_sources(root: Path) -> list[Source]:
    sources: list[Source] = []
    wan_root = root / "logs" / "h20_3_workloads_8_policies" / "results_nccl"
    wan_tgz = root / "logs" / "h20_3_workloads_8_policies.tar.gz"
    legacy_tgz = root / "logs" / "legacy_baseline_sp4_bundle_20260606.tgz"
    legacy_root = legacy_tgz.name.replace(".tgz", "")
    use_wan_tgz = not wan_root.exists() and wan_tgz.exists()
    for workload, _label in WORKLOADS:
        profile = wan_profile_dir(workload)
        for policy, _plabel in POLICIES:
            if policy == "legacy":
                legacy_profile = legacy_profile_dir("WAN", workload)
                sources.append(
                    Source(
                        model="WAN",
                        workload=workload,
                        policy=policy,
                        path=legacy_tgz,
                        member=f"{legacy_root}/results/{legacy_profile}/legacy_sp4_result.json",
                    )
                )
                continue
            if use_wan_tgz:
                sources.append(
                    Source(
                        model="WAN",
                        workload=workload,
                        policy=policy,
                        path=wan_tgz,
                        member=f"./results_nccl/{profile}/sweep_{policy}_result.json",
                    )
                )
                continue
            sources.append(
                Source(
                    model="WAN",
                    workload=workload,
                    policy=policy,
                    path=wan_root / profile / f"sweep_{policy}_result.json",
                )
            )

    qwen_tgz = root / "logs" / "qwen_battery_4gpu_argmin_bundle_20260531.tgz"
    qwen_root = qwen_tgz.name.replace(".tgz", "")
    for workload, _label in WORKLOADS:
        profile = qwen_profile_dir(workload)
        for policy, _plabel in POLICIES:
            if policy == "legacy":
                legacy_profile = legacy_profile_dir("QWEN", workload)
                sources.append(
                    Source(
                        model="QWEN",
                        workload=workload,
                        policy=policy,
                        path=legacy_tgz,
                        member=f"{legacy_root}/results/{legacy_profile}/legacy_sp4_result.json",
                    )
                )
                continue
            sources.append(
                Source(
                    model="QWEN",
                    workload=workload,
                    policy=policy,
                    path=qwen_tgz,
                    member=f"{qwen_root}/results/{profile}/sweep_{policy}_result.json",
                )
            )
    return sources


def summarize_result(source: Source, data: dict) -> dict:
    total = 0
    met = 0
    for cls in data.get("per_class", {}).values():
        total += int(cls.get("slo_defined", 0))
        met += int(cls.get("slo_met", 0))
    slo = met / total if total else math.nan
    row = {
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
    return row


def load_rows(root: Path) -> list[dict]:
    rows = []
    for source in discover_sources(root):
        rows.append(summarize_result(source, read_json(source)))
    return rows


def row_lookup(rows: list[dict]) -> dict[tuple[str, str, str], dict]:
    return {(r["model"], r["workload"], r["policy"]): r for r in rows}


def write_tables(rows: list[dict], out_prefix: Path) -> None:
    csv_path = out_prefix.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fields = list(rows[0].keys()) + ["mean_latency_norm_to_legacy"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        lookup = row_lookup(rows)
        for row in rows:
            enriched = dict(row)
            legacy = lookup[(row["model"], row["workload"], "legacy")]
            enriched["mean_latency_norm_to_legacy"] = row["latency_mean_s"] / legacy["latency_mean_s"]
            writer.writerow(enriched)

    md_path = out_prefix.with_suffix(".md")
    with md_path.open("w", encoding="utf-8") as f:
        f.write("# 4-GPU Main Result Draft\n\n")
        f.write("| model | workload | policy | SLO % | mean s | p95 s | qps | mean/legacy |\n")
        f.write("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |\n")
        lookup = row_lookup(rows)
        for model in MODELS:
            for workload, _wlabel in WORKLOADS:
                legacy = lookup[(model, workload, "legacy")]
                for policy, label in POLICIES:
                    row = lookup[(model, workload, policy)]
                    f.write(
                        f"| {model} | {workload} | {label} | "
                        f"{row['slo_attainment'] * 100:.1f} | "
                        f"{row['latency_mean_s']:.1f} | {row['latency_p95_s']:.1f} | "
                        f"{row['throughput_qps']:.3f} | "
                        f"{row['latency_mean_s'] / legacy['latency_mean_s']:.2f} |\n"
                    )


def filter_workloads(workloads: str) -> list[tuple[str, str]]:
    wanted = [item.strip() for item in workloads.split(",") if item.strip()]
    known = dict(WORKLOADS)
    missing = [item for item in wanted if item not in known]
    if missing:
        raise ValueError(f"Unknown workload(s): {missing}; valid={list(known)}")
    return [(item, known[item]) for item in wanted]


def plot_slo(rows: list[dict], out_prefix: Path, plot_workloads: list[tuple[str, str]]) -> None:
    lookup = row_lookup(rows)
    fig, axes = plt.subplots(2, len(plot_workloads), figsize=(4.95, 3.35), sharey=True)
    x = np.arange(len(POLICIES))

    for r, model in enumerate(MODELS):
        for c, (workload, wlabel) in enumerate(plot_workloads):
            ax = axes[r][c] if len(plot_workloads) > 1 else axes[r]
            values = [lookup[(model, workload, policy)]["slo_attainment"] * 100 for policy, _ in POLICIES]
            bars = ax.bar(x, values, width=0.72, color=COLORS, edgecolor="black", linewidth=0.35)
            for bar, hatch in zip(bars, HATCHES):
                bar.set_hatch(hatch)
            ax.set_ylim(0, 105)
            ax.set_title(f"{model} / {wlabel}")
            ax.grid(axis="y", color=GRID_COLOR, alpha=0.55, linewidth=0.6)
            ax.set_xticks(x)
            if r == 1:
                ax.set_xticklabels([label for _p, label in POLICIES], rotation=32, ha="right")
            else:
                ax.set_xticklabels([])
            if c == 0:
                ax.set_ylabel("SLO attainment (%)")
            for i, value in enumerate(values):
                if math.isfinite(value):
                    ax.text(i, value + 2.0, f"{value:.0f}", ha="center", va="bottom", fontsize=6)

    handles = []
    labels = []
    for color, hatch, (_policy, label) in zip(COLORS, HATCHES, POLICIES):
        patch = plt.Rectangle((0, 0), 1, 1, facecolor=color, edgecolor="black", linewidth=0.35, hatch=hatch)
        handles.append(patch)
        labels.append(label)
    fig.legend(handles, labels, frameon=False, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=[0, 0, 1, 0.95], w_pad=0.8, h_pad=0.8)
    fig.savefig(out_prefix.with_name(out_prefix.name + "_slo.pdf"), bbox_inches="tight")
    fig.savefig(out_prefix.with_name(out_prefix.name + "_slo.png"), dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_latency(rows: list[dict], out_prefix: Path, plot_workloads: list[tuple[str, str]]) -> None:
    lookup = row_lookup(rows)
    fig, axes = plt.subplots(2, len(plot_workloads), figsize=(4.95, 3.35), sharey=True)
    x = np.arange(len(POLICIES))

    for r, model in enumerate(MODELS):
        for c, (workload, wlabel) in enumerate(plot_workloads):
            ax = axes[r][c] if len(plot_workloads) > 1 else axes[r]
            legacy = lookup[(model, workload, "legacy")]["latency_mean_s"]
            values = [lookup[(model, workload, policy)]["latency_mean_s"] / legacy for policy, _ in POLICIES]
            bars = ax.bar(x, values, width=0.72, color=COLORS, edgecolor="black", linewidth=0.35)
            for bar, hatch in zip(bars, HATCHES):
                bar.set_hatch(hatch)
            ax.axhline(1.0, color=REFERENCE_COLOR, linewidth=0.8, linestyle="--")
            ax.set_ylim(0, 1.25)
            ax.set_title(f"{model} / {wlabel}")
            ax.grid(axis="y", color=GRID_COLOR, alpha=0.55, linewidth=0.6)
            ax.set_xticks(x)
            if r == 1:
                ax.set_xticklabels([label for _p, label in POLICIES], rotation=32, ha="right")
            else:
                ax.set_xticklabels([])
            if c == 0:
                ax.set_ylabel("mean latency / legacy")
            for i, value in enumerate(values):
                if math.isfinite(value):
                    ax.text(i, value + 0.035, f"{value:.2f}", ha="center", va="bottom", fontsize=6)

    handles = []
    labels = []
    for color, hatch, (_policy, label) in zip(COLORS, HATCHES, POLICIES):
        patch = plt.Rectangle((0, 0), 1, 1, facecolor=color, edgecolor="black", linewidth=0.35, hatch=hatch)
        handles.append(patch)
        labels.append(label)
    fig.legend(handles, labels, frameon=False, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=[0, 0, 1, 0.95], w_pad=0.8, h_pad=0.8)
    fig.savefig(out_prefix.with_name(out_prefix.name + "_latency_norm.pdf"), bbox_inches="tight")
    fig.savefig(out_prefix.with_name(out_prefix.name + "_latency_norm.png"), dpi=240, bbox_inches="tight")
    plt.close(fig)


def fmt_qps(value: float) -> str:
    if value >= 0.1:
        return f"{value:.2f}"
    return f"{value:.3f}"


def plot_scorecard(rows: list[dict], out_prefix: Path, plot_workloads: list[tuple[str, str]]) -> None:
    lookup = row_lookup(rows)
    scenario_keys = [(model, workload) for model in MODELS for workload, _ in plot_workloads]
    scenario_labels = [
        f"{model}\n{dict(WORKLOADS)[workload]}" for model, workload in scenario_keys
    ]
    policy_labels = [label for _policy, label in POLICIES]
    slo = np.array(
        [
            [lookup[(model, workload, policy)]["slo_attainment"] * 100 for policy, _label in POLICIES]
            for model, workload in scenario_keys
        ]
    )

    fig, ax = plt.subplots(figsize=(7.15, 3.25))
    im = ax.imshow(slo, cmap=SLO_CMAP, vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(np.arange(len(POLICIES)))
    ax.set_xticklabels(policy_labels, rotation=0)
    ax.xaxis.tick_top()
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False, pad=2)
    ax.set_yticks(np.arange(len(scenario_keys)))
    ax.set_yticklabels(scenario_labels)

    ax.set_xticks(np.arange(-0.5, len(POLICIES), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(scenario_keys), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.4)
    ax.tick_params(which="minor", bottom=False, left=False)

    for i, (model, workload) in enumerate(scenario_keys):
        best_j = int(np.nanargmax(slo[i]))
        ax.add_patch(
            plt.Rectangle(
                (best_j - 0.5, i - 0.5),
                1,
                1,
                fill=False,
                edgecolor="black",
                linewidth=1.4,
            )
        )
        for j, (policy, _label) in enumerate(POLICIES):
            row = lookup[(model, workload, policy)]
            text = (
                f"{row['slo_attainment'] * 100:.0f}%\n"
                f"m/p50 {row['latency_mean_s']:.0f}/{row['latency_p50_s']:.0f}\n"
                f"p95/99 {row['latency_p95_s']:.0f}/{row['latency_p99_s']:.0f}\n"
                f"q {fmt_qps(row['throughput_qps'])}"
            )
            t = ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                fontsize=5.8,
                color="black",
                linespacing=1.12,
            )
            t.set_path_effects([path_effects.withStroke(linewidth=1.6, foreground="white")])

    ax.set_title(
        "4-GPU main-result scorecard: color=SLO, text=latency seconds and QPS",
        pad=26,
    )
    cbar = fig.colorbar(im, ax=ax, orientation="vertical", fraction=0.025, pad=0.012)
    cbar.set_label("SLO attainment (%)")
    cbar.set_ticks([0, 25, 50, 75, 100])
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.tight_layout()
    fig.savefig(out_prefix.with_name(out_prefix.name + "_scorecard.pdf"), bbox_inches="tight")
    fig.savefig(out_prefix.with_name(out_prefix.name + "_scorecard.png"), dpi=260, bbox_inches="tight")
    plt.close(fig)


def plot_metric_grid(rows: list[dict], out_prefix: Path, plot_workloads: list[tuple[str, str]]) -> None:
    lookup = row_lookup(rows)
    scenarios = [(model, workload) for model in MODELS for workload, _ in plot_workloads]
    scenario_labels = [f"{model}\n{dict(plot_workloads)[workload]}" for model, workload in scenarios]
    metric_specs = [
        ("slo_attainment", "SLO (%)", lambda row: row["slo_attainment"] * 100.0, (0, 105)),
        ("latency_mean_s", "Mean latency (s)", lambda row: row["latency_mean_s"], None),
        ("latency_p50_s", "Median latency (s)", lambda row: row["latency_p50_s"], None),
        ("latency_p95_s", "P95 latency (s)", lambda row: row["latency_p95_s"], None),
        ("latency_p99_s", "P99 latency (s)", lambda row: row["latency_p99_s"], None),
        ("throughput_qps", "Throughput (QPS)", lambda row: row["throughput_qps"], None),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(7.1, 3.85), sharex=False)
    bar_w = 0.13
    group_x = np.arange(len(scenarios))
    offsets = (np.arange(len(POLICIES)) - (len(POLICIES) - 1) / 2.0) * bar_w

    for ax, (_metric, title, value_fn, ylim) in zip(axes.flat, metric_specs):
        for pidx, (policy, _label) in enumerate(POLICIES):
            values = [value_fn(lookup[(model, workload, policy)]) for model, workload in scenarios]
            bars = ax.bar(
                group_x + offsets[pidx],
                values,
                width=bar_w * 0.92,
                color=COLORS[pidx],
                edgecolor="black",
                linewidth=0.25,
                hatch=HATCHES[pidx],
            )
            if title.startswith("SLO"):
                for x, value in zip(group_x + offsets[pidx], values):
                    if math.isfinite(value):
                        ax.text(x, value + 2.0, f"{value:.0f}", ha="center", va="bottom", fontsize=4.9, rotation=90)
        ax.set_title(title)
        ax.grid(axis="y", color=GRID_COLOR, alpha=0.5, linewidth=0.55)
        if ylim is not None:
            ax.set_ylim(*ylim)
        else:
            ymax = max(
                value_fn(lookup[(model, workload, policy)])
                for model, workload in scenarios
                for policy, _label in POLICIES
                if math.isfinite(value_fn(lookup[(model, workload, policy)]))
            )
            ax.set_ylim(0, ymax * 1.12)
        ax.tick_params(axis="x", length=0)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    for ax in axes[1]:
        ax.set_xticks(group_x)
        ax.set_xticklabels(scenario_labels, rotation=0)
    for ax in axes[0]:
        ax.set_xticks(group_x)
        ax.set_xticklabels([])

    handles = []
    labels = []
    for color, hatch, (_policy, label) in zip(COLORS, HATCHES, POLICIES):
        patch = plt.Rectangle((0, 0), 1, 1, facecolor=color, edgecolor="black", linewidth=0.35, hatch=hatch)
        handles.append(patch)
        labels.append(label)
    fig.legend(handles, labels, frameon=False, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(rect=[0, 0, 1, 0.94], w_pad=0.8, h_pad=1.0)
    fig.savefig(out_prefix.with_name(out_prefix.name + "_metric_grid.pdf"), bbox_inches="tight")
    fig.savefig(out_prefix.with_name(out_prefix.name + "_metric_grid.png"), dpi=260, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out-prefix", type=Path, default=Path("figs/main_result_4gpu"))
    parser.add_argument(
        "--plot-workloads",
        default=",".join(DEFAULT_PLOT_WORKLOADS),
        help="Comma-separated workloads to include in plots. All workloads remain in CSV/MD.",
    )
    args = parser.parse_args()

    setup_style()
    root = args.root.resolve()
    out_prefix = args.out_prefix if args.out_prefix.is_absolute() else root / args.out_prefix
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    rows = load_rows(root)
    plot_workloads = filter_workloads(args.plot_workloads)
    write_tables(rows, out_prefix)
    plot_slo(rows, out_prefix, plot_workloads)
    plot_latency(rows, out_prefix, plot_workloads)
    plot_scorecard(rows, out_prefix, plot_workloads)
    plot_metric_grid(rows, out_prefix, plot_workloads)
    print(f"Wrote {out_prefix.with_name(out_prefix.name + '_slo.pdf')}")
    print(f"Wrote {out_prefix.with_name(out_prefix.name + '_slo.png')}")
    print(f"Wrote {out_prefix.with_name(out_prefix.name + '_latency_norm.pdf')}")
    print(f"Wrote {out_prefix.with_name(out_prefix.name + '_latency_norm.png')}")
    print(f"Wrote {out_prefix.with_name(out_prefix.name + '_scorecard.pdf')}")
    print(f"Wrote {out_prefix.with_name(out_prefix.name + '_scorecard.png')}")
    print(f"Wrote {out_prefix.with_name(out_prefix.name + '_metric_grid.pdf')}")
    print(f"Wrote {out_prefix.with_name(out_prefix.name + '_metric_grid.png')}")
    print(f"Wrote {out_prefix.with_suffix('.csv')}")
    print(f"Wrote {out_prefix.with_suffix('.md')}")


if __name__ == "__main__":
    main()
