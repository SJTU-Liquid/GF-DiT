#!/usr/bin/env python3
"""Visualize request traces used by the scheduling experiments."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import tarfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from paper_style import ALL_SERIES_COLOR, GRID_COLOR, PHASE_COLORS, REQUEST_CLASS_COLORS, TEXT_MUTED


CLASS_ORDER = ["S", "M", "L"]
CLASS_COLORS = REQUEST_CLASS_COLORS


@dataclass(frozen=True)
class WorkloadSource:
    model: str
    profile: str
    label: str
    source: str
    member: str | None = None


def read_json_source(source: WorkloadSource) -> dict:
    path = Path(source.source)
    if source.member is None:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    with tarfile.open(path, "r:*") as tf:
        member = tf.extractfile(source.member)
        if member is None:
            raise FileNotFoundError(f"{source.member} not found in {path}")
        with io.TextIOWrapper(member, encoding="utf-8") as f:
            return json.load(f)


def discover_sources(root: Path) -> list[WorkloadSource]:
    wan_dir = root / "logs" / "h20_3_workloads_8_policies" / "workloads"
    wan_tgz = root / "logs" / "h20_3_workloads_8_policies.tar.gz"
    use_wan_tgz = not wan_dir.exists() and wan_tgz.exists()
    qwen_tgz = root / "logs" / "qwen_battery_4gpu_argmin_bundle_20260531.tgz"
    if not qwen_tgz.exists():
        qwen_tgz = root / "logs" / "qwen_battery_planB_bundle_20260529.tgz"

    sources = [
        WorkloadSource(
            "WAN",
            "short",
            "WAN short",
            str(wan_tgz if use_wan_tgz else wan_dir / "h20bat_short_tl1.20_a2.0_2.5_3.5_seed42.json"),
            "./workloads/h20bat_short_tl1.20_a2.0_2.5_3.5_seed42.json" if use_wan_tgz else None,
        ),
        WorkloadSource(
            "WAN",
            "inflight",
            "WAN inflight",
            str(wan_tgz if use_wan_tgz else wan_dir / "h20bat_inflight_burst_tl0.90_a2.0_2.5_3.5_seed42.json"),
            "./workloads/h20bat_inflight_burst_tl0.90_a2.0_2.5_3.5_seed42.json" if use_wan_tgz else None,
        ),
        WorkloadSource(
            "WAN",
            "fg",
            "WAN fg",
            str(wan_tgz if use_wan_tgz else wan_dir / "h20bat_foreground_burst_tl1.10_a2.0_2.5_3.5_seed42.json"),
            "./workloads/h20bat_foreground_burst_tl1.10_a2.0_2.5_3.5_seed42.json" if use_wan_tgz else None,
        ),
    ]

    qwen_root = qwen_tgz.stem.removesuffix(".tar")
    # tarfile names for .tgz use the directory basename without the .tgz suffix.
    if qwen_root.endswith(".tar"):
        qwen_root = qwen_root[:-4]
    qwen_root = qwen_tgz.name.replace(".tgz", "")

    sources += [
        WorkloadSource(
            "QWEN",
            "short",
            "QWEN short",
            str(qwen_tgz),
            f"{qwen_root}/workloads/qwenbat_short_tl1.20_seed42.json",
        ),
        WorkloadSource(
            "QWEN",
            "inflight",
            "QWEN inflight",
            str(qwen_tgz),
            f"{qwen_root}/workloads/qwenbat_inflight_burst_tl0.70_seed42.json",
        ),
        WorkloadSource(
            "QWEN",
            "fg",
            "QWEN fg",
            str(qwen_tgz),
            f"{qwen_root}/workloads/qwenbat_foreground_burst_tl0.90_seed42.json",
        ),
    ]
    return sources


def request_rows(workload: dict) -> list[dict]:
    rows = []
    for req in workload["requests"]:
        arrival_s = req["arrival_ms"] / 1000.0
        service_s = req["profiled_optimal_latency_ms"] / 1000.0
        deadline_s = req["deadline_ms"] / 1000.0
        client_deadline_s = req.get("client_deadline_ms", req["deadline_ms"]) / 1000.0
        rows.append(
            {
                "arrival_s": arrival_s,
                "class": req["request_class"],
                "service_s": service_s,
                "deadline_slack_s": deadline_s - arrival_s,
                "client_deadline_slack_s": client_deadline_s - arrival_s,
                "latent_seq_len": req.get("latent_seq_len", math.nan),
                "height": req.get("height", math.nan),
                "width": req.get("width", math.nan),
                "frames": req.get("num_frames", math.nan),
                "preemptible": bool(req.get("preemptible", False)),
            }
        )
    return sorted(rows, key=lambda x: x["arrival_s"])


def phase_at(phases: list[dict], t: float) -> str:
    for phase in phases:
        if phase["start_s"] <= t < phase["end_s"]:
            return phase["name"]
    return "outside"


def binned_counts(rows: list[dict], duration_s: float, bin_s: float) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    bins = np.arange(0.0, duration_s + bin_s, bin_s)
    centers = (bins[:-1] + bins[1:]) / 2.0
    counts = {}
    for klass in CLASS_ORDER:
        values = [row["arrival_s"] for row in rows if row["class"] == klass]
        counts[klass], _ = np.histogram(values, bins=bins)
    return centers, counts


def add_phase_background(ax, phases: list[dict], ymax: float | None = None) -> None:
    top = ymax if ymax is not None else 1.0
    for idx, phase in enumerate(phases):
        color = PHASE_COLORS[idx % len(PHASE_COLORS)]
        ax.axvspan(phase["start_s"] / 60.0, phase["end_s"] / 60.0, color=color, zorder=0)
        mid = (phase["start_s"] + phase["end_s"]) / 120.0
        label = phase["name"].replace("_", " ")
        ax.text(mid, top, label, ha="center", va="top", fontsize=8, color=TEXT_MUTED, clip_on=True)


def make_overview(workloads: list[tuple[WorkloadSource, dict, list[dict]]], outdir: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 6.4), sharey=False)
    profiles = ["short", "inflight", "fg"]
    models = ["WAN", "QWEN"]
    lookup = {(src.model, src.profile): (src, workload, rows) for src, workload, rows in workloads}

    for r, model in enumerate(models):
        for c, profile in enumerate(profiles):
            ax = axes[r][c]
            src, workload, rows = lookup[(model, profile)]
            meta = workload["metadata"]
            duration_s = float(meta["duration_s"])
            bin_s = 30.0 if duration_s <= 1200 else 60.0
            centers, counts = binned_counts(rows, duration_s, bin_s)
            rates = [counts[k] * 60.0 / bin_s for k in CLASS_ORDER]
            ymax = max(1.0, sum(rates).max() * 1.18)
            add_phase_background(ax, meta.get("phases", []), ymax)
            bottom = np.zeros_like(centers)
            for klass, rate in zip(CLASS_ORDER, rates):
                ax.bar(
                    centers / 60.0,
                    rate,
                    width=bin_s / 60.0 * 0.92,
                    bottom=bottom,
                    color=CLASS_COLORS[klass],
                    label=klass,
                    edgecolor="white",
                    linewidth=0.3,
                    zorder=2,
                )
                bottom += rate
            title = f"{src.label}: n={len(rows)}, load={meta.get('offered_load_realized', math.nan):.2f}"
            ax.set_title(title, fontsize=10)
            ax.set_xlim(0, duration_s / 60.0)
            ax.set_ylim(0, ymax)
            ax.grid(axis="y", color=GRID_COLOR, alpha=0.5, linewidth=0.6)
            if c == 0:
                ax.set_ylabel("arrivals/min")
            if r == 1:
                ax.set_xlabel("time (min)")
            if r == 0 and c == 2:
                ax.legend(frameon=False, loc="upper right", ncol=3, fontsize=8)

    fig.suptitle("Workload arrival-rate timeline by request class", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(outdir / "workload_trace_overview.pdf")
    fig.savefig(outdir / "workload_trace_overview.png", dpi=220)
    plt.close(fig)


def make_paper_overview(workloads: list[tuple[WorkloadSource, dict, list[dict]]], outdir: Path) -> None:
    """Compact single-column view of the workloads used in the main results."""
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 8.5,
            "axes.labelsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(3.35, 2.0), sharey=False)
    profiles = ["short", "fg"]
    models = ["WAN", "QWEN"]
    lookup = {(src.model, src.profile): (src, workload, rows) for src, workload, rows in workloads}

    for r, model in enumerate(models):
        for c, profile in enumerate(profiles):
            ax = axes[r][c]
            _, workload, rows = lookup[(model, profile)]
            meta = workload["metadata"]
            duration_s = float(meta["duration_s"])
            bin_s = 30.0 if duration_s <= 1200 else 60.0
            centers, counts = binned_counts(rows, duration_s, bin_s)
            rates = [counts[k] * 60.0 / bin_s for k in CLASS_ORDER]
            ymax = max(1.0, sum(rates).max() * 1.15)
            for idx, phase in enumerate(meta.get("phases", [])):
                ax.axvspan(
                    phase["start_s"] / 60.0,
                    phase["end_s"] / 60.0,
                    color=PHASE_COLORS[idx % len(PHASE_COLORS)],
                    zorder=0,
                )
            bottom = np.zeros_like(centers)
            for klass, rate in zip(CLASS_ORDER, rates):
                ax.bar(
                    centers / 60.0,
                    rate,
                    width=bin_s / 60.0 * 0.92,
                    bottom=bottom,
                    color=CLASS_COLORS[klass],
                    label=klass,
                    edgecolor="white",
                    linewidth=0.2,
                    zorder=2,
                )
                bottom += rate
            workload_name = "short" if profile == "short" else "FG burst"
            ax.set_title(f"{model} {workload_name}")
            ax.set_xlim(0, duration_s / 60.0)
            ax.set_ylim(0, ymax)
            ax.grid(axis="y", color=GRID_COLOR, alpha=0.5, linewidth=0.35)
            if c == 0:
                ax.set_ylabel("arrivals/min")
            if r == 1:
                ax.set_xlabel("time (min)")
            for spine in ["top", "right"]:
                ax.spines[spine].set_visible(False)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.01))
    fig.tight_layout(rect=[0, 0, 1, 0.95], h_pad=0.65, w_pad=0.65)
    fig.savefig(outdir / "workload_trace_paper.pdf", bbox_inches="tight")
    fig.savefig(outdir / "workload_trace_paper.png", dpi=270, bbox_inches="tight")
    plt.close(fig)


def make_scatter(workloads: list[tuple[WorkloadSource, dict, list[dict]]], outdir: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 6.4), sharey=False)
    profiles = ["short", "inflight", "fg"]
    models = ["WAN", "QWEN"]
    lookup = {(src.model, src.profile): (src, workload, rows) for src, workload, rows in workloads}

    for r, model in enumerate(models):
        for c, profile in enumerate(profiles):
            ax = axes[r][c]
            src, workload, rows = lookup[(model, profile)]
            meta = workload["metadata"]
            duration_s = float(meta["duration_s"])
            max_service = max(row["service_s"] for row in rows) if rows else 1.0
            add_phase_background(ax, meta.get("phases", []), max_service * 1.08)
            for klass in CLASS_ORDER:
                xs = [row["arrival_s"] / 60.0 for row in rows if row["class"] == klass]
                ys = [row["service_s"] for row in rows if row["class"] == klass]
                slack = [row["deadline_slack_s"] / max(row["service_s"], 1e-9) for row in rows if row["class"] == klass]
                sizes = [18 + 18 * min(max(s - 1.0, 0.0), 5.0) / 5.0 for s in slack]
                ax.scatter(xs, ys, s=sizes, color=CLASS_COLORS[klass], alpha=0.82, label=klass, zorder=3)
            ax.set_title(src.label, fontsize=10)
            ax.set_xlim(0, duration_s / 60.0)
            ax.set_ylim(0, max_service * 1.12)
            ax.grid(axis="y", color=GRID_COLOR, alpha=0.5, linewidth=0.6)
            if c == 0:
                ax.set_ylabel("profiled service time (s)")
            if r == 1:
                ax.set_xlabel("arrival time (min)")
            if r == 0 and c == 2:
                ax.legend(frameon=False, loc="upper right", ncol=3, fontsize=8)

    fig.suptitle("Individual requests: arrival time vs profiled service time", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(outdir / "workload_trace_scatter.pdf")
    fig.savefig(outdir / "workload_trace_scatter.png", dpi=220)
    plt.close(fig)


def make_distributions(workloads: list[tuple[WorkloadSource, dict, list[dict]]], outdir: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 6.4), sharex=False)
    profiles = ["short", "inflight", "fg"]
    models = ["WAN", "QWEN"]
    lookup = {(src.model, src.profile): (src, workload, rows) for src, workload, rows in workloads}

    for r, model in enumerate(models):
        for c, profile in enumerate(profiles):
            ax = axes[r][c]
            src, _workload, rows = lookup[(model, profile)]
            arrivals = np.array([row["arrival_s"] for row in rows])
            gaps = np.diff(arrivals)
            if len(gaps):
                for klass in CLASS_ORDER:
                    class_arrivals = np.array([row["arrival_s"] for row in rows if row["class"] == klass])
                    class_gaps = np.diff(class_arrivals)
                    if len(class_gaps):
                        sorted_gaps = np.sort(class_gaps)
                        y = 1.0 - np.arange(1, len(sorted_gaps) + 1) / len(sorted_gaps)
                        ax.step(sorted_gaps, y, where="post", color=CLASS_COLORS[klass], label=klass)
                sorted_gaps = np.sort(gaps)
                y = 1.0 - np.arange(1, len(sorted_gaps) + 1) / len(sorted_gaps)
                ax.step(sorted_gaps, y, where="post", color=ALL_SERIES_COLOR, linewidth=1.4, label="all")
            ax.set_title(src.label, fontsize=10)
            ax.set_xscale("symlog", linthresh=1.0)
            ax.set_ylim(-0.02, 1.02)
            ax.grid(color=GRID_COLOR, alpha=0.45, linewidth=0.6)
            if c == 0:
                ax.set_ylabel("CCDF")
            if r == 1:
                ax.set_xlabel("inter-arrival gap (s)")
            if r == 0 and c == 2:
                ax.legend(frameon=False, loc="upper right", ncol=4, fontsize=8)

    fig.suptitle("Inter-arrival gap distribution", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(outdir / "workload_trace_interarrival_ccdf.pdf")
    fig.savefig(outdir / "workload_trace_interarrival_ccdf.png", dpi=220)
    plt.close(fig)


def quantile(values: list[float], q: float) -> float:
    if not values:
        return math.nan
    return float(np.quantile(np.array(values, dtype=float), q))


def summarize(src: WorkloadSource, workload: dict, rows: list[dict]) -> dict:
    meta = workload["metadata"]
    duration_s = float(meta["duration_s"])
    active_end = max((row["arrival_s"] for row in rows), default=0.0)
    counts = Counter(row["class"] for row in rows)
    per_min_counts = []
    if duration_s > 0:
        bins = np.arange(0.0, duration_s + 60.0, 60.0)
        hist, _ = np.histogram([row["arrival_s"] for row in rows], bins=bins)
        active_hist = hist[hist > 0]
        per_min_counts = active_hist.tolist()
    mean_active_rate = float(np.mean(per_min_counts)) if per_min_counts else 0.0
    max_rate = float(np.max(per_min_counts)) if per_min_counts else 0.0
    burst_ratio = max_rate / mean_active_rate if mean_active_rate > 0 else math.nan
    gaps = np.diff(np.array([row["arrival_s"] for row in rows], dtype=float)).tolist()
    slacks = [row["deadline_slack_s"] / row["service_s"] for row in rows]
    service_by_class = defaultdict(list)
    for row in rows:
        service_by_class[row["class"]].append(row["service_s"])

    return {
        "model": src.model,
        "profile": src.profile,
        "requests": len(rows),
        "duration_min": duration_s / 60.0,
        "last_arrival_min": active_end / 60.0,
        "offered_load_expected": meta.get("offered_load_expected", math.nan),
        "offered_load_realized": meta.get("offered_load_realized", math.nan),
        "S": counts.get("S", 0),
        "M": counts.get("M", 0),
        "L": counts.get("L", 0),
        "S_frac": counts.get("S", 0) / len(rows) if rows else math.nan,
        "M_frac": counts.get("M", 0) / len(rows) if rows else math.nan,
        "L_frac": counts.get("L", 0) / len(rows) if rows else math.nan,
        "mean_arrivals_per_min_active_bins": mean_active_rate,
        "max_arrivals_per_min": max_rate,
        "burst_ratio": burst_ratio,
        "gap_p50_s": quantile(gaps, 0.50),
        "gap_p95_s": quantile(gaps, 0.95),
        "service_S_s": quantile(service_by_class["S"], 0.50),
        "service_M_s": quantile(service_by_class["M"], 0.50),
        "service_L_s": quantile(service_by_class["L"], 0.50),
        "deadline_slack_over_service_p50": quantile(slacks, 0.50),
        "deadline_slack_over_service_p05": quantile(slacks, 0.05),
    }


def write_summary(workloads: list[tuple[WorkloadSource, dict, list[dict]]], outdir: Path) -> None:
    rows = [summarize(src, workload, reqs) for src, workload, reqs in workloads]
    csv_path = outdir / "workload_trace_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    md_path = outdir / "workload_trace_summary.md"
    cols = [
        "model",
        "profile",
        "requests",
        "duration_min",
        "offered_load_realized",
        "S",
        "M",
        "L",
        "burst_ratio",
        "gap_p50_s",
        "gap_p95_s",
        "service_S_s",
        "service_M_s",
        "service_L_s",
        "deadline_slack_over_service_p05",
    ]
    with md_path.open("w", encoding="utf-8") as f:
        f.write("# Workload trace summary\n\n")
        f.write("| " + " | ".join(cols) + " |\n")
        f.write("| " + " | ".join(["---"] * len(cols)) + " |\n")
        for row in rows:
            values = []
            for col in cols:
                value = row[col]
                if isinstance(value, float):
                    values.append("" if math.isnan(value) else f"{value:.2f}")
                else:
                    values.append(str(value))
            f.write("| " + " | ".join(values) + " |\n")
        f.write("\n## Phase schedules\n\n")
        for src, workload, _reqs in workloads:
            f.write(f"### {src.label}\n\n")
            f.write("| phase | start min | end min | arrivals/min | mix |\n")
            f.write("| --- | ---: | ---: | ---: | --- |\n")
            for phase in workload["metadata"].get("phases", []):
                mix = ", ".join(f"{k}:{phase['mix'].get(k, 0):.2f}" for k in CLASS_ORDER)
                f.write(
                    f"| {phase['name']} | {phase['start_s'] / 60.0:.2f} | "
                    f"{phase['end_s'] / 60.0:.2f} | {phase['arrival_rate_per_min']:.2f} | {mix} |\n"
                )
            f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--outdir", type=Path, default=Path("figs"))
    args = parser.parse_args()

    root = args.root.resolve()
    outdir = (root / args.outdir).resolve() if not args.outdir.is_absolute() else args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    workloads = []
    for source in discover_sources(root):
        workload = read_json_source(source)
        workloads.append((source, workload, request_rows(workload)))

    make_overview(workloads, outdir)
    make_paper_overview(workloads, outdir)
    make_scatter(workloads, outdir)
    make_distributions(workloads, outdir)
    write_summary(workloads, outdir)
    print(f"Wrote workload trace figures and summaries to {outdir}")


if __name__ == "__main__":
    main()
