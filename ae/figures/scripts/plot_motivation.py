#!/usr/bin/env python3
"""Generate the motivation figure from H20 profiling and policy logs."""

from __future__ import annotations

import argparse
import csv
import io
import json
import tarfile
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from paper_style import COMPONENT_COLORS, GRID_COLOR, POLICY_COLORS, REFERENCE_COLOR, SHAPE_COLORS


GROUPS = [1, 2, 4, 8]
SHAPE_SERIES = [
    (1950.0, "480x832/17f"),
    (2420.0, "720x720/17f"),
    (5070.0, "480x832/49f"),
    (18480.0, "720x1280/81f"),
]
COLORS = {
    "DiT": COMPONENT_COLORS["DiT"],
    "VAE": COMPONENT_COLORS["VAE"],
    "Text encoder": COMPONENT_COLORS["Text encoder"],
    "SP1": SHAPE_COLORS[0],
    "SP2": SHAPE_COLORS[1],
    "SP4": SHAPE_COLORS[2],
    "SP8": SHAPE_COLORS[3],
    "480x480/17f": SHAPE_COLORS[0],
    "480x832/17f": SHAPE_COLORS[1],
    "720x720/17f": SHAPE_COLORS[2],
    "480x832/49f": SHAPE_COLORS[3],
    "720x1280/81f": SHAPE_COLORS[4],
    "Ideal": COMPONENT_COLORS["Ideal"],
}
OBJECTIVE_POLICY_COLORS = {
    "SRTF-SP1": POLICY_COLORS["SRTF-SP1"],
    "SRTF-SP4": POLICY_COLORS["SRTF-SP4"],
    "FCFS-SP1": POLICY_COLORS["FCFS-SP1"],
    "SJF-SP1": POLICY_COLORS["SJF-SP1"],
}
POLICY_CHOICE_SCENARIOS = [
    "WAN short",
    "WAN FG burst",
    "QWEN short",
    "QWEN FG burst",
]
POLICY_OBJECTIVES = [
    ("SLO", "overall", "slo_pct", "max", lambda value: f"{value:.0f}%"),
    ("P95", "overall", "p95_s", "min", lambda value: f"{value:.0f}s"),
    ("S p95", "S", "p95_s", "min", lambda value: f"{value:.0f}s"),
    ("M SLO", "M", "slo_pct", "max", lambda value: f"{value:.0f}%"),
]
MARKERS = {
    "DiT": "o",
    "VAE": "s",
    "Text encoder": "D",
    "SP1": "o",
    "SP2": "s",
    "SP4": "^",
    "SP8": "D",
    "480x480/17f": "o",
    "480x832/17f": "s",
    "720x720/17f": "^",
    "480x832/49f": "P",
    "720x1280/81f": "<",
}


def median_by_x(samples: list[list[float]]) -> dict[float, float]:
    grouped: dict[float, list[float]] = defaultdict(list)
    for x, y in samples:
        grouped[float(x)].append(float(y))
    return {x: float(np.median(values)) for x, values in grouped.items()}


def read_json_from_tar(tar_path: Path, member: str) -> dict:
    with tarfile.open(tar_path, "r:*") as tf:
        with tf.extractfile(member) as f:
            if f is None:
                raise FileNotFoundError(member)
            return json.load(f)


def read_csv_from_tar(tar_path: Path, member: str) -> list[dict[str, str]]:
    with tarfile.open(tar_path, "r:*") as tf:
        extracted = tf.extractfile(member)
        if extracted is None:
            raise FileNotFoundError(member)
        with io.TextIOWrapper(extracted, encoding="utf-8") as f:
            return list(csv.DictReader(f))


def load_objective_metric_rows(tar_path: Path) -> list[dict[str, str]]:
    return read_csv_from_tar(tar_path, "objective_policy_sim_bundle/data/all_metrics_long.csv")


def load_vae_scaling(tar_path: Path, shape: tuple[int, int, int]) -> dict[int, float]:
    height, width, frames = shape
    members = {
        1: "./results/native.json",
        2: "./results/tiled_pp2.json",
        4: "./results/tiled_pp4.json",
        8: "./results/tiled_pp8.json",
    }
    latencies = {}
    for pp, member in members.items():
        data = read_json_from_tar(tar_path, member)
        for sample in data["samples"]:
            if (
                sample["height"] == height
                and sample["width"] == width
                and sample["num_frames"] == frames
            ):
                latencies[pp] = float(sample["latency_ms_median"])
                break
        else:
            raise KeyError(f"shape {shape} not found in {member}")
    return latencies


def load_dit_scaling(cost_dir: Path, latent_seq_len: float) -> dict[int, float]:
    latencies = {}
    for group in GROUPS:
        path = cost_dir / f"tp1_ulysses{group}_ring1_cfg1.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        stage = data["stages"]["dit_step_chunk"]
        by_x = median_by_x(stage["samples"])
        latencies[group] = by_x[float(latent_seq_len)]
    return latencies


def load_text_encoder_scaling(cost_dir: Path) -> dict[int, float]:
    path = cost_dir / "tp1_ulysses1_ring1_cfg1.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    values = [float(y) for _x, y in data["stages"]["text_encode"]["samples"]]
    median = float(np.median(values))
    return {group: median for group in GROUPS}


def load_shape_scaling(cost_dir: Path) -> dict[int, dict[float, float]]:
    by_sp = {}
    common = None
    for sp in GROUPS:
        path = cost_dir / f"tp1_ulysses{sp}_ring1_cfg1.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        medians = median_by_x(data["stages"]["dit_step_chunk"]["samples"])
        by_sp[sp] = medians
        keys = set(medians)
        common = keys if common is None else common & keys
    if common is None:
        return by_sp
    # SP8 was profiled on the production-relevant shape range. Restrict to
    # shapes present for all SP degrees so curves are directly comparable.
    common_sorted = sorted(common)
    return {sp: {x: by_sp[sp][x] for x in common_sorted} for sp in GROUPS}


def write_data_tables(
    out_prefix: Path,
    dit_stage: dict[int, float],
    vae_stage: dict[int, float],
    text_stage: dict[int, float],
    shape_scaling: dict[int, dict[float, float]],
    objective_rows: list[dict[str, str]],
) -> None:
    policy_choices = compute_policy_choices(objective_rows)
    with out_prefix.with_name(out_prefix.name + "_data.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["panel", "series", "x", "latency_ms", "normalized_latency", "objective", "policy"])
        for series, values in [("DiT", dit_stage), ("VAE", vae_stage), ("Text encoder", text_stage)]:
            base = values[1]
            for group in GROUPS:
                writer.writerow(["stage", series, group, values[group], values[group] / base, "", ""])
        for latent, label in SHAPE_SERIES:
            base = shape_scaling[1][latent]
            for sp in GROUPS:
                latency = shape_scaling[sp][latent]
                writer.writerow(["shape", label, sp, latency, latency / base, "", ""])
        for choice in policy_choices:
            writer.writerow(
                [
                    "policy_choice",
                    choice["scenario"],
                    choice["value"],
                    "",
                    "",
                    choice["objective"],
                    choice["policy_label"],
                ]
            )

    with out_prefix.with_name(out_prefix.name + "_data.md").open("w", encoding="utf-8") as f:
        f.write("# Motivation Figure Data\n\n")
        f.write("## Stage Scaling\n\n")
        f.write("| series | group | latency ms | normalized |\n")
        f.write("| --- | ---: | ---: | ---: |\n")
        for series, values in [("DiT", dit_stage), ("VAE", vae_stage), ("Text encoder", text_stage)]:
            base = values[1]
            for group in GROUPS:
                f.write(f"| {series} | {group} | {values[group]:.1f} | {values[group] / base:.3f} |\n")
        f.write("\n## DiT Shape Scaling\n\n")
        f.write("| shape | SP | latency ms | normalized |\n")
        f.write("| --- | ---: | ---: | ---: |\n")
        for latent, label in SHAPE_SERIES:
            base = shape_scaling[1][latent]
            for sp in GROUPS:
                latency = shape_scaling[sp][latent]
                f.write(f"| {label} | {sp} | {latency:.1f} | {latency / base:.3f} |\n")
        f.write("\n## Policy Choice\n\n")
        f.write("| scenario | objective | best policy | value | scope | metric |\n")
        f.write("| --- | --- | --- | ---: | --- | --- |\n")
        for choice in policy_choices:
            objective_text = choice["objective"].replace("\n", " ")
            f.write(
                f"| {choice['scenario'].replace(chr(10), ' ')} | {objective_text} | "
                f"{choice['policy_label']} | {choice['value_text']} | "
                f"{choice['scope']} | {choice['metric']} |\n"
            )


def panel_stage(
    ax,
    dit_stage: dict[int, float],
    vae_stage: dict[int, float],
    text_stage: dict[int, float],
) -> None:
    for label, values in [("DiT", dit_stage), ("VAE", vae_stage), ("Text encoder", text_stage)]:
        base = values[1]
        ys = [values[g] / base for g in GROUPS]
        linestyle = "--" if label == "Text encoder" else "-"
        ax.plot(
            GROUPS,
            ys,
            marker=MARKERS[label],
            linewidth=1.7,
            markersize=5,
            color=COLORS[label],
            linestyle=linestyle,
            label=label,
        )
    ax.axhline(1.0, color=REFERENCE_COLOR, linewidth=0.8, linestyle=":")
    ax.set_title("(a) Stage scaling")
    ax.set_xlabel("execution group size")
    ax.set_ylabel("normalized latency")
    ax.set_xticks(GROUPS)
    ax.set_xlim(0.5, 8.5)
    ax.set_ylim(0.0, 1.25)
    ax.grid(True, color=GRID_COLOR, alpha=0.45, linewidth=0.6)
    ax.legend(frameon=False, loc="lower left", fontsize=6.5, handlelength=1.2, labelspacing=0.3, borderpad=0.2, handletextpad=0.4)


def panel_shape(ax, shape_scaling: dict[int, dict[float, float]]) -> None:
    x = np.arange(len(GROUPS))
    ax.plot(
        x,
        [1.0 / group for group in GROUPS],
        color=COLORS["Ideal"],
        linestyle="--",
        linewidth=1.2,
        label="ideal 1/SP",
    )
    for latent, label in SHAPE_SERIES:
        base = shape_scaling[1][latent]
        ys = [shape_scaling[sp][latent] / base for sp in GROUPS]
        ax.plot(
            x,
            ys,
            marker=MARKERS[label],
            linewidth=1.7,
            markersize=4.5,
            color=COLORS[label],
            label=label,
        )
    ax.axhline(1.0, color=REFERENCE_COLOR, linewidth=0.8, linestyle=":")
    ax.set_title("(b) Shape-dependent SP")
    ax.set_xlabel("sequence parallel degree")
    ax.set_xticks(x)
    ax.set_xticklabels([str(group) for group in GROUPS])
    ax.set_xlim(-0.15, x[-1] + 0.15)
    ax.set_ylim(0.0, 1.2)
    ax.grid(True, color=GRID_COLOR, alpha=0.45, linewidth=0.6)
    legend = ax.legend(frameon=True, loc="upper right", ncol=2, fontsize=6.5, columnspacing=0.6, handlelength=1.1, handletextpad=0.4, labelspacing=0.25, borderpad=0.3)
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("none")
    legend.get_frame().set_alpha(0.85)


def compute_policy_choices(objective_rows: list[dict[str, str]]) -> list[dict]:
    rows_by_cell: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in objective_rows:
        rows_by_cell[(row["workload"], row["scope"])].append(row)

    choices = []
    for scenario in POLICY_CHOICE_SCENARIOS:
        for objective, scope, metric, direction, fmt_value in POLICY_OBJECTIVES:
            candidates = rows_by_cell[(scenario, scope)]
            key_fn = lambda row: float(row[metric])
            best = max(candidates, key=key_fn) if direction == "max" else min(candidates, key=key_fn)
            value = key_fn(best)
            choices.append(
                {
                    "scenario": scenario.replace(" FG burst", "\nFG burst").replace(" short", "\nshort"),
                    "objective": objective,
                    "scope": scope,
                    "metric": metric,
                    "policy": best["policy"],
                    "policy_label": best["policy"],
                    "value": value,
                    "value_text": fmt_value(value),
                }
            )
    return choices


def panel_policy_choice(ax, objective_rows: list[dict[str, str]]) -> None:
    choices = compute_policy_choices(objective_rows)
    by_cell = {
        (choice["scenario"], choice["objective"]): choice
        for choice in choices
    }
    scenarios = [
        scenario.replace(" FG burst", "\nFG burst").replace(" short", "\nshort")
        for scenario in POLICY_CHOICE_SCENARIOS
    ]
    objectives = [objective for objective, _scope, _metric, _direction, _fmt in POLICY_OBJECTIVES]

    ax.set_xlim(-0.5, len(objectives) - 0.5)
    ax.set_ylim(len(scenarios) - 0.5, -0.5)
    ax.set_xticks(range(len(objectives)))
    ax.set_xticklabels(objectives)
    ax.xaxis.tick_top()
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False, length=0, pad=2)
    ax.set_yticks(range(len(scenarios)))
    ax.set_yticklabels(scenarios)
    ax.set_title("(c) Objective-dependent winners")

    # Winner identity is carried by the cell color (figure-level legend on
    # top); the cell itself shows only the metric value, which keeps the text
    # large without overflowing the cells.
    for y, scenario in enumerate(scenarios):
        for x, objective in enumerate(objectives):
            choice = by_cell[(scenario, objective)]
            color = OBJECTIVE_POLICY_COLORS[choice["policy_label"]]
            ax.add_patch(
                plt.Rectangle(
                    (x - 0.5, y - 0.5),
                    1,
                    1,
                    facecolor=color,
                    edgecolor="white",
                    linewidth=1.2,
                    alpha=0.9,
                )
            )
            ax.text(
                x,
                y,
                choice["value_text"],
                ha="center",
                va="center",
                fontsize=8,
                color="black",
            )

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(objectives), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(scenarios), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=color, edgecolor="white")
        for _label, color in OBJECTIVE_POLICY_COLORS.items()
    ]
    ax.legend(
        handles,
        list(OBJECTIVE_POLICY_COLORS.keys()),
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.015),
        ncol=2,
        columnspacing=0.8,
        handlelength=1.0,
        handletextpad=0.35,
        fontsize=8,
    )


def make_figure(
    out_prefix: Path,
    dit_stage: dict[int, float],
    vae_stage: dict[int, float],
    shape_scaling: dict[int, dict[float, float]],
    text_stage: dict[int, float],
    objective_rows: list[dict[str, str]],
) -> None:
    # Designed at final size (\textwidth ~7.0 in): 1 pt here is 1 pt on the
    # page, so 8-9 pt text prints at 8-9 pt.
    plt.rcParams.update(
        {
            "font.size": 8.5,
            "axes.titlesize": 9,
            "axes.labelsize": 8.5,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(
        1, 3, figsize=(7.0, 2.15), gridspec_kw={"width_ratios": [0.95, 1.35, 1.3]}
    )
    panel_stage(axes[0], dit_stage, vae_stage, text_stage)
    panel_shape(axes[1], shape_scaling)
    panel_policy_choice(axes[2], objective_rows)
    fig.tight_layout(w_pad=1.0)
    fig.savefig(out_prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_prefix.with_suffix(".png"), dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vae-bundle", type=Path, default=Path("logs/vae_pp_bundle_complete.tgz"))
    parser.add_argument("--cost-dir", type=Path, default=Path("logs/wan22-ti2v-5b-h20-cost-model"))
    parser.add_argument("--objective-bundle", type=Path, default=Path("logs/objective_policy_sim_bundle.tgz"))
    parser.add_argument("--out-prefix", type=Path, default=Path("figs/motivation"))
    parser.add_argument("--stage-latent-seq-len", type=float, default=18480.0)
    parser.add_argument("--vae-shape", default="720x1280x81")
    args = parser.parse_args()

    height, width, frames = [int(part) for part in args.vae_shape.split("x")]
    dit_stage = load_dit_scaling(args.cost_dir, args.stage_latent_seq_len)
    vae_stage = load_vae_scaling(args.vae_bundle, (height, width, frames))
    text_stage = load_text_encoder_scaling(args.cost_dir)
    shape_scaling = load_shape_scaling(args.cost_dir)
    objective_rows = load_objective_metric_rows(args.objective_bundle)

    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    make_figure(args.out_prefix, dit_stage, vae_stage, shape_scaling, text_stage, objective_rows)
    write_data_tables(args.out_prefix, dit_stage, vae_stage, text_stage, shape_scaling, objective_rows)
    print(f"Wrote {args.out_prefix.with_suffix('.pdf')}")
    print(f"Wrote {args.out_prefix.with_suffix('.png')}")
    print(f"Wrote {args.out_prefix.with_name(args.out_prefix.name + '_data.csv')}")
    print(f"Wrote {args.out_prefix.with_name(args.out_prefix.name + '_data.md')}")


if __name__ == "__main__":
    main()
