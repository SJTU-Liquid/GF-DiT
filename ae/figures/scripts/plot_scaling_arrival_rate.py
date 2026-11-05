#!/usr/bin/env python3
"""Plot arrival-rate/load scaling from the H20 4-GPU scaling bundle."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import re
import tarfile
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler

from paper_style import GRID_COLOR, POLICY_COLORS


POLICIES = [
    ("edf_best_fit", "EDF", POLICY_COLORS["edf_best_fit"], "o"),
    ("srtf_sp1", "SRTF-SP1", POLICY_COLORS["srtf_sp1"], "s"),
]
MODEL_LABELS = {"wan": "WAN", "qwen": "QWEN"}
METRICS = [
    ("slo_attainment", "SLO (%)", lambda row: row["slo_attainment"] * 100.0, (0, 105), False),
    ("latency_mean_s", "Mean latency (s)", lambda row: row["latency_mean_s"], None, False),
    ("latency_p95_s", "P95 latency (s)", lambda row: row["latency_p95_s"], None, False),
    ("throughput_qps", "Throughput (QPS)", lambda row: row["throughput_qps"], None, False),
]


def setup_style() -> None:
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    plt.rcParams["axes.prop_cycle"] = cycler("color", [p[2] for p in POLICIES])
    small = 8
    plt.rc("font", size=small)
    plt.rc("axes", titlesize=small)
    plt.rc("axes", labelsize=small)
    plt.rc("xtick", labelsize=small)
    plt.rc("ytick", labelsize=small)
    plt.rc("legend", fontsize=small)
    plt.rc("figure", titlesize=small)


def read_json(tf: tarfile.TarFile, member: str) -> dict:
    extracted = tf.extractfile(member)
    if extracted is None:
        raise FileNotFoundError(member)
    with io.TextIOWrapper(extracted, encoding="utf-8") as f:
        return json.load(f)


def summarize_result(data: dict) -> dict:
    total = 0
    met = 0
    for cls in data.get("per_class", {}).values():
        total += int(cls.get("slo_defined", 0))
        met += int(cls.get("slo_met", 0))
    return {
        "slo_attainment": met / total if total else math.nan,
        "latency_mean_s": float(data.get("latency_mean", math.nan)),
        "latency_p50_s": float(data.get("latency_p50", data.get("latency_median", math.nan))),
        "latency_p95_s": float(data.get("latency_p95", math.nan)),
        "latency_p99_s": float(data.get("latency_p99", math.nan)),
        "throughput_qps": float(data.get("throughput_qps", math.nan)),
        "completed_requests": int(data.get("completed_requests", 0)),
        "failed_requests": int(data.get("failed_requests", 0)),
    }


def workload_member(model: str, target_load: str) -> str:
    prefix = "wan" if model == "wan" else "qwen"
    stem = "wanscale4" if model == "wan" else "qwenscale4"
    return f"{prefix}_scaling4/{stem}_short_tl{target_load}_seed42.json"


def load_rows(bundle: Path) -> list[dict]:
    rows = []
    result_re = re.compile(
        r"(?P<model>wan|qwen)_scaling4_real/"
        r"(?P<profile>(?:wan|qwen)scale4_short_tl(?P<load>[0-9.]+)_seed42)/"
        r"sweep_(?P<policy>edf_best_fit|srtf_sp1)_result\.json$"
    )
    with tarfile.open(bundle, "r:*") as tf:
        members = tf.getnames()
        for member in members:
            match = result_re.match(member)
            if not match:
                continue
            model = match.group("model")
            load_text = match.group("load")
            result = summarize_result(read_json(tf, member))
            workload = read_json(tf, workload_member(model, load_text))
            metadata = workload.get("metadata", {})
            rows.append(
                {
                    "gpu": "H20",
                    "gpu_count": int(metadata.get("num_gpus", 4)),
                    "model": MODEL_LABELS[model],
                    "workload": metadata.get("profile", "short"),
                    "target_load": float(load_text),
                    "offered_load_realized": float(metadata.get("offered_load_realized", math.nan)),
                    "requests": len(workload.get("requests", [])),
                    "policy": match.group("policy"),
                    **result,
                    "result_member": member,
                    "workload_member": workload_member(model, load_text),
                }
            )
    return sorted(rows, key=lambda r: (r["model"], r["policy"], r["target_load"]))


def write_tables(rows: list[dict], out_prefix: Path) -> None:
    fields = [
        "gpu",
        "gpu_count",
        "model",
        "workload",
        "target_load",
        "offered_load_realized",
        "requests",
        "policy",
        "slo_attainment",
        "latency_mean_s",
        "latency_p50_s",
        "latency_p95_s",
        "latency_p99_s",
        "throughput_qps",
        "completed_requests",
        "failed_requests",
        "result_member",
        "workload_member",
    ]
    with out_prefix.with_suffix(".csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})

    policy_labels = {policy: label for policy, label, _color, _marker in POLICIES}
    with out_prefix.with_suffix(".md").open("w", encoding="utf-8") as f:
        f.write("# Arrival-Rate Scaling\n\n")
        f.write("| model | target load | realized load | requests | policy | SLO % | mean s | p50 s | p95 s | qps |\n")
        f.write("| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |\n")
        for row in rows:
            f.write(
                f"| {row['model']} | {row['target_load']:.1f} | {row['offered_load_realized']:.2f} | "
                f"{row['requests']} | {policy_labels[row['policy']]} | "
                f"{row['slo_attainment'] * 100:.1f} | {row['latency_mean_s']:.1f} | "
                f"{row['latency_p50_s']:.1f} | {row['latency_p95_s']:.1f} | "
                f"{row['throughput_qps']:.3f} |\n"
            )


def plot_scaling(rows: list[dict], out_prefix: Path, x_field: str) -> None:
    models = sorted({row["model"] for row in rows})
    if not models:
        raise SystemExit("No scaling result rows found")
    if len(models) != 1:
        raise SystemExit("2x2 single-column layout expects exactly one model")
    model_rows = [row for row in rows if row["model"] == models[0]]
    # Single-column figure: 2x2 metric grid sized for \linewidth (~3.35 in).
    fig, axes = plt.subplots(2, 2, figsize=(3.35, 2.0), squeeze=False, sharex=True)

    for idx, (_metric, title, value_fn, ylim, log_y) in enumerate(METRICS):
        ax = axes[idx // 2][idx % 2]
        for policy, label, color, marker in POLICIES:
            series = sorted([row for row in model_rows if row["policy"] == policy], key=lambda row: row[x_field])
            if not series:
                continue
            xs = [row[x_field] for row in series]
            ys = [value_fn(row) for row in series]
            ax.plot(xs, ys, color=color, marker=marker, linewidth=1.6, markersize=4.2, label=label)
        ax.set_title(title, pad=2.5)
        if ylim is not None:
            ax.set_ylim(*ylim)
        if log_y:
            ax.set_yscale("log")
        ax.grid(True, color=GRID_COLOR, alpha=0.45, linewidth=0.55)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    for ax in axes[-1]:
        ax.set_xlabel("target load" if x_field == "target_load" else "realized offered load")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="upper center", ncol=len(POLICIES), bbox_to_anchor=(0.5, 1.05))
    fig.tight_layout(rect=[0, 0, 1, 0.94], w_pad=0.6, h_pad=0.7)
    fig.savefig(out_prefix.with_name(out_prefix.name + ".pdf"), bbox_inches="tight")
    fig.savefig(out_prefix.with_name(out_prefix.name + ".png"), dpi=270, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, default=Path("logs/scaling_arrival_rate_bundle.tgz"))
    parser.add_argument("--out-prefix", type=Path, default=Path("figs/scaling_arrival_rate"))
    parser.add_argument("--x", choices=["target_load", "offered_load_realized"], default="target_load")
    args = parser.parse_args()

    setup_style()
    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.bundle)
    write_tables(rows, args.out_prefix)
    plot_scaling(rows, args.out_prefix, args.x)
    print(f"Wrote {args.out_prefix.with_suffix('.pdf')}")
    print(f"Wrote {args.out_prefix.with_suffix('.png')}")
    print(f"Wrote {args.out_prefix.with_suffix('.csv')}")
    print(f"Wrote {args.out_prefix.with_suffix('.md')}")


if __name__ == "__main__":
    main()
