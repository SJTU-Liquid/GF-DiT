#!/usr/bin/env python3
"""Plot NCCL subgroup setup cost and one representative sim-vs-real scenario."""

from __future__ import annotations

import argparse
import csv
import io
import json
import tarfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from paper_style import GRID_COLOR, PALETTE, SYSTEM_COLORS


POLICY_ORDER = [
    "edf_best_fit",
    "srtf_sp1",
    "srtf_sp4",
    "fcfs_sp1",
    "fcfs_sp4",
]
POLICY_LABELS = {
    "edf_best_fit": "EDF",
    "srtf_sp1": "SRTF-SP1",
    "srtf_sp4": "SRTF-SPmax",
    "fcfs_sp1": "FCFS-SP1",
    "fcfs_sp4": "FCFS-SP4",
}


def setup_style() -> None:
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 8,
            "axes.labelsize": 8,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    )


def load_subgroup_rows(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["size_sweep"]


def plot_subgroup_setup(rows: list[dict], out_prefix: Path) -> None:
    sizes = [row["size"] for row in rows]
    x = np.arange(len(sizes))
    width = 0.24
    series = [
        ("new_group", "newgroup_cpu_ms_max", PALETTE["green"], ""),
        ("first collective", "first_coll_ms_mean", PALETTE["yellow"], "///"),
        ("warm collective", "warm_coll_ms_max", PALETTE["blue"], "\\\\\\"),
    ]

    fig, ax = plt.subplots(figsize=(3.35, 2.05))
    for offset, (label, field, color, hatch) in zip([-width, 0.0, width], series):
        ax.bar(
            x + offset,
            [row[field] for row in rows],
            width,
            label=label,
            color=color,
            edgecolor="black",
            linewidth=0.35,
            hatch=hatch,
        )
    ax.set_yscale("log")
    ax.set_ylabel("latency (ms, log scale)")
    ax.set_xlabel("NCCL subgroup size")
    ax.set_xticks(x)
    ax.set_xticklabels(sizes)
    ax.grid(axis="y", which="both", color=GRID_COLOR, alpha=0.5, linewidth=0.55)
    ax.legend(frameon=False, loc="upper left", ncol=1)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_prefix.with_suffix(".png"), dpi=260, bbox_inches="tight")
    plt.close(fig)


def load_sim_vs_real_rows(bundle: Path) -> list[dict[str, str]]:
    # The aligned bundle (request-id-aligned simulator replay) is the source
    # used by the paper; the unaligned bundle has larger per-policy errors.
    member = "sim_vs_real_aligned/h20/sim_vs_real.csv"
    if "aligned" not in bundle.name:
        member = "sim_vs_real/h20/sim_vs_real.csv"
    with tarfile.open(bundle, "r:*") as tf:
        extracted = tf.extractfile(member)
        if extracted is None:
            raise FileNotFoundError(member)
        with io.TextIOWrapper(extracted, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    return [
        row
        for row in rows
        if row["row"] == "WAN fg" and row["scope"] == "overall" and row["policy"] in POLICY_ORDER
    ]


def plot_sim_vs_real(rows: list[dict[str, str]], out_prefix: Path) -> None:
    lookup = {row["policy"]: row for row in rows}
    x = np.arange(len(POLICY_ORDER))
    width = 0.36
    sim = [float(lookup[policy]["sim_slo"]) for policy in POLICY_ORDER]
    real = [float(lookup[policy]["real_slo"]) for policy in POLICY_ORDER]

    fig, ax = plt.subplots(figsize=(3.35, 1.9))
    ax.bar(
        x - width / 2,
        sim,
        width,
        label="Simulator",
        color=SYSTEM_COLORS["Simulator"],
        edgecolor="black",
        linewidth=0.35,
    )
    ax.bar(
        x + width / 2,
        real,
        width,
        label="Real H20",
        color=SYSTEM_COLORS["Real H20"],
        edgecolor="black",
        linewidth=0.35,
        hatch="///",
    )
    for xs, vals in ((x - width / 2, sim), (x + width / 2, real)):
        for xi, v in zip(xs, vals):
            ax.text(xi, v + 2.0, f"{v:.0f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("SLO attainment (%)")
    ax.set_ylim(0, 116)
    ax.set_xticks(x)
    ax.set_xticklabels([POLICY_LABELS[policy] for policy in POLICY_ORDER], rotation=35, ha="right")
    ax.grid(axis="y", color=GRID_COLOR, alpha=0.5, linewidth=0.55)
    ax.legend(frameon=False, loc="upper right", ncol=1, borderaxespad=0.1, handletextpad=0.5, labelspacing=0.3)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_prefix.with_suffix(".png"), dpi=260, bbox_inches="tight")
    plt.close(fig)


def write_sim_vs_real_csv(rows: list[dict[str, str]], out: Path) -> None:
    lookup = {row["policy"]: row for row in rows}
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["policy", "sim_slo_pct", "real_slo_pct", "absolute_error_pp"])
        for policy in POLICY_ORDER:
            row = lookup[policy]
            sim = float(row["sim_slo"])
            real = float(row["real_slo"])
            writer.writerow([policy, sim, real, abs(sim - real)])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--subgroup-json",
        type=Path,
        default=Path("logs/nccl_subgroup_cost_h20_sizesweep.json"),
    )
    parser.add_argument("--sim-bundle", type=Path, default=Path("logs/sim_vs_real_aligned.tgz"))
    parser.add_argument("--out-dir", type=Path, default=Path("figs"))
    args = parser.parse_args()

    setup_style()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    subgroup_prefix = args.out_dir / "nccl_subgroup_setup"
    sim_prefix = args.out_dir / "sim_vs_real_h20_fg"

    subgroup_rows = load_subgroup_rows(args.subgroup_json)
    plot_subgroup_setup(subgroup_rows, subgroup_prefix)

    sim_rows = load_sim_vs_real_rows(args.sim_bundle)
    plot_sim_vs_real(sim_rows, sim_prefix)
    write_sim_vs_real_csv(sim_rows, sim_prefix.with_suffix(".csv"))

    print(f"Wrote {subgroup_prefix.with_suffix('.pdf')}")
    print(f"Wrote {sim_prefix.with_suffix('.pdf')}")
    print(f"Wrote {sim_prefix.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
