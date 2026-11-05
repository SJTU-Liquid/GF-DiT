#!/usr/bin/env python3
"""Redraw Figure 5: slot-state designs for overlapping dynamic groups."""

from __future__ import annotations

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch


# Publication-friendly, editable text in the PDF.
mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "font.size": 8.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

GREEN = "#2E8B57"
GREEN_FILL = "#F2F8F4"
ORANGE = "#E07B39"
ORANGE_DARK = "#C84D13"
ORANGE_LIGHT = "#E99A57"
ORANGE_FILL = "#FFF7EF"
BLUE = "#2B6CB0"
BLUE_FILL = "#F3F7FC"
RED = "#B91C1C"
INK = "#111111"
MUTED = "#5B6470"
WHITE = "#FFFFFF"

FIG_W, FIG_H = 7.0, 2.6
PANEL_Y, PANEL_H = 0.035, 2.53
PANEL_W = 2.29
PANEL_X = (0.025, 2.355, 4.685)


def local(panel_x: float, x: float, y: float) -> tuple[float, float]:
    """Convert panel-local coordinates (0..1) to figure data coordinates."""
    return panel_x + PANEL_W * x, PANEL_Y + PANEL_H * y


def text(ax, x, y, s, *, size=8.0, color=INK, weight="normal", style="normal",
         ha="center", va="center", zorder=10, linespacing=1.05):
    return ax.text(
        x,
        y,
        s,
        fontsize=size,
        color=color,
        fontweight=weight,
        fontstyle=style,
        ha=ha,
        va=va,
        zorder=zorder,
        linespacing=linespacing,
    )


def rounded_box(ax, xy, width, height, *, edge, face=WHITE, lw=1.05,
                radius=0.045, pad=0.012, zorder=2):
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle=f"round,pad={pad},rounding_size={radius}",
        linewidth=lw,
        edgecolor=edge,
        facecolor=face,
        joinstyle="round",
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def arrow(ax, start, end, *, color, lw=1.15, mutation=8, zorder=5):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=mutation,
        linewidth=lw,
        color=color,
        shrinkA=0,
        shrinkB=0,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def draw_panel(ax, panel_x: float, color: str, fill: str, title: str):
    rounded_box(
        ax,
        (panel_x, PANEL_Y),
        PANEL_W,
        PANEL_H,
        edge=color,
        face=fill,
        lw=1.35,
        radius=0.055,
        pad=0.0,
        zorder=0,
    )
    text(
        ax,
        *local(panel_x, 0.5, 0.947),
        title,
        size=9.4,
        weight="bold",
    )


def rank_node(ax, center, rank: int, *, edge=INK, lw=1.05, radius=0.088,
              face=WHITE, text_color=INK, zorder=8):
    ax.add_patch(
        Circle(
            center,
            radius=radius,
            facecolor=face,
            edgecolor=edge,
            linewidth=lw,
            zorder=zorder,
        )
    )
    text(ax, *center, str(rank), size=9.1, color=text_color, zorder=zorder + 1)


def draw_rank_header(ax, panel_x: float, *, node_edge=INK):
    text(
        ax,
        *local(panel_x, 0.06, 0.825),
        "Ranks",
        size=8.7,
        weight="bold",
        style="italic",
        ha="left",
    )
    for rank, xpos in enumerate((0.33, 0.51, 0.69, 0.87)):
        rank_node(ax, local(panel_x, xpos, 0.825), rank, edge=node_edge)


def verdict(ax, panel_x: float, *, color: str, label: str, width=0.72, size=9.0):
    x0 = 0.5 - width / 2
    bx, by = local(panel_x, x0, 0.045)
    bw, bh = PANEL_W * width, PANEL_H * 0.102
    rounded_box(
        ax,
        (bx, by),
        bw,
        bh,
        edge=color,
        face=WHITE,
        lw=1.2,
        radius=0.042,
        pad=0.0,
        zorder=4,
    )
    text(
        ax,
        bx + bw / 2,
        by + bh / 2,
        label,
        size=size,
        color=color,
        weight="bold",
    )


def draw_group_transition(ax, panel_x: float, *, color: str):
    y = 0.705
    text(
        ax,
        *local(panel_x, 0.20, y),
        "GA = {0,1,2}",
        size=8.3,
        color=color,
        weight="bold",
    )
    arrow(
        ax,
        local(panel_x, 0.405, y),
        local(panel_x, 0.595, y),
        color=color,
        lw=1.35,
        mutation=8.5,
        zorder=8,
    )
    text(
        ax,
        *local(panel_x, 0.80, y),
        "GB = {1,2,3}",
        size=8.3,
        color=color,
        weight="bold",
    )


def draw_overlap_network(ax, panel_x: float, *, color: str, y_center=0.515,
                         shared_color=None, shared_lw=1.8):
    """Two tidy triangular group hulls with the shared (1,2) pair."""
    p0 = local(panel_x, 0.12, y_center)
    p1 = local(panel_x, 0.50, y_center + 0.060)
    p2 = local(panel_x, 0.50, y_center - 0.060)
    p3 = local(panel_x, 0.88, y_center)

    for cx in (0.35, 0.65):
        center = local(panel_x, cx, y_center)
        ax.add_patch(
            Ellipse(
                center,
                width=PANEL_W * 0.54,
                height=PANEL_H * 0.235,
                facecolor="none",
                edgecolor=ORANGE_LIGHT if color == ORANGE else color,
                linewidth=1.15 if color == ORANGE else 1.0,
                alpha=0.95,
                zorder=1,
            )
        )

    # GA triangle and GB triangle. The common vertical edge is drawn last.
    for a, b in ((p0, p1), (p0, p2), (p1, p3), (p2, p3)):
        ax.plot(
            (a[0], b[0]),
            (a[1], b[1]),
            color=color,
            linewidth=1.2,
            solid_capstyle="round",
            zorder=3,
        )
    edge_color = shared_color if shared_color is not None else ORANGE_DARK
    ax.plot(
        (p1[0], p2[0]),
        (p1[1], p2[1]),
        color=edge_color,
        linewidth=shared_lw,
        solid_capstyle="round",
        zorder=4,
    )

    for rank, point in enumerate((p0, p1, p2, p3)):
        rank_node(ax, point, rank, radius=0.080, lw=1.0)


def draw_panel_a(ax):
    px = PANEL_X[0]
    draw_panel(ax, px, GREEN, GREEN_FILL, "(a) Separate slots per group")
    draw_rank_header(ax, px)

    text(
        ax,
        *local(px, 0.34, 0.705),
        "Many dynamic\nlogical groups",
        size=8.2,
        color=GREEN,
        weight="bold",
        linespacing=1.0,
    )
    text(
        ax,
        *local(px, 0.805, 0.705),
        "Barrier slots\n(per group)",
        size=8.0,
        color=GREEN,
        weight="bold",
        linespacing=1.0,
    )

    row_centers = (0.555, 0.425, 0.295)
    group_specs = (("G1 = {0,1}", 0, 1), ("G2 = {0,2}", 0, 2), ("G3 = {1,2}", 1, 2))
    for y, (label, r0, r1) in zip(row_centers, group_specs):
        bx, by = local(px, 0.035, y - 0.052)
        rounded_box(
            ax,
            (bx, by),
            PANEL_W * 0.645,
            PANEL_H * 0.104,
            edge=GREEN,
            face=WHITE,
            lw=0.95,
            radius=0.034,
            pad=0.0,
            zorder=2,
        )
        text(
            ax,
            *local(px, 0.065, y),
            label,
            size=8.2,
            ha="left",
        )
        q0 = local(px, 0.505, y)
        q1 = local(px, 0.625, y)
        ax.plot(
            (q0[0], q1[0]),
            (q0[1], q1[1]),
            color=GREEN,
            linewidth=1.2,
            solid_capstyle="round",
            zorder=5,
        )
        rank_node(ax, q0, r0, radius=0.057, lw=0.95)
        rank_node(ax, q1, r1, radius=0.057, lw=0.95)

    # One dedicated barrier-state column, visually matched to all three rows.
    sx, sy = local(px, 0.735, 0.241)
    sw, sh = PANEL_W * 0.235, PANEL_H * 0.366
    rounded_box(
        ax,
        (sx, sy),
        sw,
        sh,
        edge=GREEN,
        face=WHITE,
        lw=1.05,
        radius=0.050,
        pad=0.0,
        zorder=2,
    )
    for sep_y in (0.360, 0.490):
        y_abs = local(px, 0, sep_y)[1]
        ax.plot(
            (sx + 0.055, sx + sw - 0.055),
            (y_abs, y_abs),
            color=GREEN,
            linewidth=0.65,
            alpha=0.45,
            zorder=3,
        )
    for idx, y in enumerate(row_centers, start=1):
        text(ax, *local(px, 0.8525, y), f"Slot G{idx}", size=8.2, weight="bold")

    text(ax, *local(px, 0.355, 0.200), "\N{HORIZONTAL ELLIPSIS}", size=11.0)
    verdict(ax, px, color=GREEN, label="Correct but combinatorial", width=0.90, size=8.8)


def draw_panel_b(ax):
    px = PANEL_X[1]
    draw_panel(ax, px, ORANGE, ORANGE_FILL, "(b) Shared slots per group")
    draw_rank_header(ax, px)
    draw_group_transition(ax, px, color=ORANGE)
    draw_overlap_network(ax, px, color=ORANGE, y_center=0.515, shared_color=RED)

    # A compact causal timeline: the second group reaches the same state slot
    # before the first group's token has been consumed. Stacking the events
    # makes the temporal order explicit without reducing the 8 pt body text.
    tx, ty = local(px, 0.04, 0.155)
    tw, th = PANEL_W * 0.92, PANEL_H * 0.235
    rounded_box(
        ax,
        (tx, ty),
        tw,
        th,
        edge=ORANGE,
        face=WHITE,
        lw=1.0,
        radius=0.040,
        pad=0.0,
        zorder=2,
    )
    for y0, label in (
        (0.298, "GA barrier: (1,2) writes Slot 0"),
        (0.166, "GB barrier: (1,2) reuses Slot 0"),
    ):
        bx, by = local(px, 0.075, y0)
        bw, bh = PANEL_W * 0.85, PANEL_H * 0.057
        rounded_box(
            ax,
            (bx, by),
            bw,
            bh,
            edge=ORANGE,
            face=ORANGE_FILL,
            lw=0.85,
            radius=0.026,
            pad=0.0,
            zorder=4,
        )
        text(
            ax,
            bx + bw / 2,
            by + bh / 2,
            label,
            size=8.0,
        )

    arrow(
        ax,
        local(px, 0.165, 0.296),
        local(px, 0.165, 0.273),
        color=ORANGE_DARK,
        lw=1.05,
        mutation=7.5,
        zorder=6,
    )
    arrow(
        ax,
        local(px, 0.165, 0.235),
        local(px, 0.165, 0.224),
        color=ORANGE_DARK,
        lw=1.05,
        mutation=7.5,
        zorder=6,
    )
    marker = local(px, 0.165, 0.254)
    ax.add_patch(Circle(marker, 0.064, facecolor=WHITE, edgecolor=RED, linewidth=1.25, zorder=7))
    text(ax, *marker, "\N{MULTIPLICATION SIGN}", size=10.0, color=RED, weight="bold", zorder=8)
    text(
        ax,
        *local(px, 0.235, 0.254),
        "overwrite before consume",
        size=8.0,
        color=RED,
        weight="bold",
        ha="left",
    )

    verdict(ax, px, color=ORANGE, label="Unsafe shared slots", width=0.68)


def draw_panel_c(ax):
    px = PANEL_X[2]
    draw_panel(ax, px, BLUE, BLUE_FILL, "(c) Shared slots per edge")
    draw_rank_header(ax, px)
    draw_group_transition(ax, px, color=BLUE)
    draw_overlap_network(
        ax,
        px,
        color=BLUE,
        y_center=0.550,
        shared_color=GREEN,
        shared_lw=4.0,
    )

    text(
        ax,
        *local(px, 0.50, 0.388),
        "Shared edge (1,2)\nflips across groups",
        size=8.2,
        color=BLUE,
        weight="bold",
        linespacing=0.90,
    )

    # Hero strip: exact group-to-slot mapping plus phase dots that alternate.
    sx, sy = local(px, 0.035, 0.158)
    sw, sh = PANEL_W * 0.93, PANEL_H * 0.180
    rounded_box(
        ax,
        (sx, sy),
        sw,
        sh,
        edge=BLUE,
        face=WHITE,
        lw=1.1,
        radius=0.045,
        pad=0.0,
        zorder=2,
    )
    for split in (0.345, 0.655):
        x_abs = local(px, split, 0)[0]
        ax.plot(
            (x_abs, x_abs),
            (sy + 0.055, sy + sh - 0.055),
            color=BLUE,
            linewidth=0.65,
            alpha=0.22,
            zorder=3,
        )

    event_xs = (0.18, 0.50, 0.82)
    event_labels = (
        "GA \N{RIGHTWARDS ARROW}\nSlot 0",
        "GB \N{RIGHTWARDS ARROW}\nSlot 1",
        "next \N{RIGHTWARDS ARROW}\nSlot 0",
    )
    phase_colors = (BLUE, GREEN, BLUE)
    for xpos, label, phase_color in zip(event_xs, event_labels, phase_colors):
        text(
            ax,
            *local(px, xpos, 0.291),
            label,
            size=8.0,
            color=BLUE,
            weight="bold",
            linespacing=0.90,
        )
        center = local(px, xpos, 0.216)
        ax.add_patch(Circle(center, 0.052, facecolor=WHITE, edgecolor=INK, linewidth=0.85, zorder=6))
        ax.add_patch(Circle(center, 0.033, facecolor=phase_color, edgecolor="none", zorder=7))

    for x0, x1 in ((0.215, 0.465), (0.535, 0.785)):
        arrow(
            ax,
            local(px, x0, 0.216),
            local(px, x1, 0.216),
            color=MUTED,
            lw=0.9,
            mutation=7.0,
            zorder=5,
        )

    verdict(ax, px, color=BLUE, label="Scalable and precise", width=0.68)


def main():
    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor=WHITE)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, FIG_W)
    ax.set_ylim(0, FIG_H)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    draw_panel_a(ax)
    draw_panel_b(ax)
    draw_panel_c(ax)

    fig.savefig(
        "fig5_redrawn.pdf",
        bbox_inches="tight",
        pad_inches=0.015,
        facecolor=WHITE,
    )
    fig.savefig(
        "fig5_redrawn.png",
        dpi=250,
        bbox_inches="tight",
        pad_inches=0.015,
        facecolor=WHITE,
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
