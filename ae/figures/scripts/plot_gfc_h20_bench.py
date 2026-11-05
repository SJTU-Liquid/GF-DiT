#!/usr/bin/env python3
"""Plot GFC vs NCCL collective latency from A100 and H20 bundles."""

from __future__ import annotations

import argparse
import csv
import re
import tarfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from paper_style import GRID_COLOR, GROUP_BACKGROUND_COLORS, REFERENCE_COLOR, SYSTEM_COLORS


COLLECTIVES = [("all2all", "A2A"), ("allgather", "AG")]
DEFAULT_SIZES = "16K,1M,4M,16M,64M"
COLORS = {"GFC": SYSTEM_COLORS["GFC"], "NCCL": SYSTEM_COLORS["NCCL"]}
HATCHES = {"GFC": "", "NCCL": "///"}


def read_tar_member(tar_path: Path, member: str) -> str:
    with tarfile.open(tar_path, "r:*") as tf:
        extracted = tf.extractfile(member)
        if extracted is None:
            raise FileNotFoundError(f"{member} not found in {tar_path}")
        return extracted.read().decode("utf-8", errors="replace")


def parse_a100_log(text: str) -> list[dict]:
    rows = []
    pattern = re.compile(
        r"(?P<collective>all2all|allgather)\s+"
        r"(?P<bytes>\d+)\s+config:[^\n]*?"
        r"latency\s+p50=(?P<gfc>[0-9.]+)us[^\n]*?"
        r"nccl_reference\s+p50=(?P<nccl>[0-9.]+)us"
    )
    for match in pattern.finditer(text):
        rows.append(
            {
                "platform": "A100 x8",
                "collective": match.group("collective"),
                "bytes": int(match.group("bytes")),
                "gfc_p50_us": float(match.group("gfc")),
                "nccl_p50_us": float(match.group("nccl")),
            }
        )
    return rows


def parse_h20_log(text: str) -> list[dict]:
    rows = []
    pattern = re.compile(
        r"(?P<collective>all2all|allgather)\s+"
        r"bytes=(?P<bytes>\d+)\s+"
        r"GFC_lat=(?P<gfc>[0-9.]+)\s+"
        r"GFC_ker=(?P<kernel>[0-9.]+)\s+"
        r"NCCL=(?P<nccl>[0-9.]+)"
    )
    for match in pattern.finditer(text):
        rows.append(
            {
                "platform": "H20 x4",
                "collective": match.group("collective"),
                "bytes": int(match.group("bytes")),
                "gfc_p50_us": float(match.group("gfc")),
                "nccl_p50_us": float(match.group("nccl")),
            }
        )
    return rows


def load_rows(a100_bundle: Path, h20_bundle: Path) -> list[dict]:
    a100_text = read_tar_member(a100_bundle, "gfc_coll_bundle/cudagraph_full_sweep_batch10_w3_i15.out")
    h20_text = read_tar_member(h20_bundle, "h20_4card_bundle/cudagraph_4card_batch10_w3_i15.log")
    rows = parse_a100_log(a100_text) + parse_h20_log(h20_text)
    if not rows:
        raise SystemExit("No collective benchmark rows parsed")
    return sorted(rows, key=lambda row: (row["platform"], row["collective"], row["bytes"]))


def parse_size(value: str) -> int:
    cleaned = value.strip().lower().replace("ib", "")
    if cleaned.endswith("k"):
        return int(float(cleaned[:-1]) * 1024)
    if cleaned.endswith("m"):
        return int(float(cleaned[:-1]) * 1024 * 1024)
    return int(cleaned)


def human_size(size_bytes: int) -> str:
    if size_bytes >= 1024 * 1024:
        return f"{size_bytes // (1024 * 1024)}M"
    if size_bytes >= 1024:
        return f"{size_bytes // 1024}K"
    return str(size_bytes)


def selected_sizes(rows: list[dict], size_arg: str) -> list[int]:
    available = {row["bytes"] for row in rows}
    if size_arg.strip().lower() == "all":
        return sorted(available)
    sizes = [parse_size(item) for item in size_arg.split(",") if item.strip()]
    missing = [human_size(size) for size in sizes if size not in available]
    if missing:
        raise ValueError(f"Requested size(s) not found in bundles: {missing}")
    return sizes


def write_csv(rows: list[dict], sizes: list[int], out: Path) -> None:
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "platform",
                "collective",
                "bytes",
                "size",
                "gfc_p50_us",
                "nccl_p50_us",
                "gfc_over_nccl",
                "nccl_over_gfc",
            ],
        )
        writer.writeheader()
        for row in rows:
            if row["bytes"] not in sizes:
                continue
            enriched = dict(row)
            enriched["size"] = human_size(row["bytes"])
            enriched["gfc_over_nccl"] = row["gfc_p50_us"] / row["nccl_p50_us"]
            enriched["nccl_over_gfc"] = row["nccl_p50_us"] / row["gfc_p50_us"]
            writer.writerow(enriched)


def write_markdown(rows: list[dict], sizes: list[int], out: Path) -> None:
    lookup = {(row["platform"], row["collective"], row["bytes"]): row for row in rows}
    with out.open("w", encoding="utf-8") as f:
        f.write("# GFC vs NCCL Collective Latency\n\n")
        f.write("CUDA-graph p50 latency, microseconds.\n\n")
        f.write("| platform | collective | size | GFC us | NCCL us | GFC/NCCL |\n")
        f.write("| --- | --- | ---: | ---: | ---: | ---: |\n")
        for platform in ["A100 x8", "H20 x4"]:
            for collective, _short in COLLECTIVES:
                for size in sizes:
                    row = lookup[(platform, collective, size)]
                    f.write(
                        f"| {platform} | {collective} | {human_size(size)} | "
                        f"{row['gfc_p50_us']:.1f} | {row['nccl_p50_us']:.1f} | "
                        f"{row['gfc_p50_us'] / row['nccl_p50_us']:.2f} |\n"
                    )


def setup_style() -> None:
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    # Designed at the single-column width it is placed at, so points here
    # are points on the page (PPoPP requires >= 8 pt inside figures).
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 8.5,
            "axes.labelsize": 8,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    )


def plot_platform(ax, rows: list[dict], platform: str, sizes: list[int]) -> None:
    lookup = {(row["collective"], row["bytes"]): row for row in rows if row["platform"] == platform}
    group_labels = []
    gfc_values = []
    nccl_values = []
    group_colors = []
    for collective, short in COLLECTIVES:
        for size in sizes:
            row = lookup[(collective, size)]
            group_labels.append(f"{short}\n{human_size(size)}")
            gfc_values.append(row["gfc_p50_us"])
            nccl_values.append(row["nccl_p50_us"])
            group_colors.append(GROUP_BACKGROUND_COLORS[collective])

    x = np.arange(len(group_labels))
    width = 0.36
    for xpos, color in zip(x, group_colors):
        ax.axvspan(xpos - 0.5, xpos + 0.5, color=color, zorder=0)

    ax.bar(
        x - width / 2,
        gfc_values,
        width,
        label="GFC",
        color=COLORS["GFC"],
        edgecolor="black",
        linewidth=0.3,
        hatch=HATCHES["GFC"],
        zorder=2,
    )
    ax.bar(
        x + width / 2,
        nccl_values,
        width,
        label="NCCL",
        color=COLORS["NCCL"],
        edgecolor="black",
        linewidth=0.3,
        hatch=HATCHES["NCCL"],
        zorder=2,
    )
    ax.set_yscale("log")
    # Plain decade labels: 10^n mathtext renders the exponent at ~0.7x the
    # base size, which lands below the 8 pt floor after placement scaling.
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.yaxis.get_major_formatter().set_scientific(False)
    ax.set_title(platform, loc="left", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(group_labels)
    ax.grid(axis="y", which="both", color=GRID_COLOR, alpha=0.45, linewidth=0.55)
    for boundary in [len(sizes) - 0.5]:
        ax.axvline(boundary, color=REFERENCE_COLOR, linewidth=0.7)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def make_plot(rows: list[dict], sizes: list[int], out_prefix: Path) -> None:
    setup_style()
    fig, axes = plt.subplots(2, 1, figsize=(3.4, 2.05), sharex=True)
    fig.supylabel("p50 latency (us)", fontsize=8)
    plot_platform(axes[0], rows, "A100 x8", sizes)
    plot_platform(axes[1], rows, "H20 x4", sizes)
    axes[0].legend(frameon=False, loc="upper left", ncol=2, bbox_to_anchor=(0.0, 1.18), columnspacing=0.8, handlelength=1.2, handletextpad=0.4)
    axes[1].set_xlabel("collective / per-rank message size")
    fig.tight_layout(h_pad=0.55)
    fig.savefig(out_prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_prefix.with_suffix(".png"), dpi=260, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a100-bundle", type=Path, default=Path("logs/gfc_a100_collective_bench.tgz"))
    parser.add_argument("--h20-bundle", type=Path, default=Path("logs/gfc_h20_4card_collective_bench.tgz"))
    parser.add_argument("--out-prefix", type=Path, default=Path("figs/gfc_collective_latency"))
    parser.add_argument(
        "--sizes",
        default=DEFAULT_SIZES,
        help='Comma-separated sizes such as "16K,1M,4M", or "all".',
    )
    args = parser.parse_args()

    rows = load_rows(args.a100_bundle, args.h20_bundle)
    sizes = selected_sizes(rows, args.sizes)
    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    make_plot(rows, sizes, args.out_prefix)
    write_csv(rows, sizes, args.out_prefix.with_suffix(".csv"))
    write_markdown(rows, sizes, args.out_prefix.with_suffix(".md"))
    print(f"Wrote {args.out_prefix.with_suffix('.pdf')}")
    print(f"Wrote {args.out_prefix.with_suffix('.png')}")
    print(f"Wrote {args.out_prefix.with_suffix('.csv')}")
    print(f"Wrote {args.out_prefix.with_suffix('.md')}")


if __name__ == "__main__":
    main()
