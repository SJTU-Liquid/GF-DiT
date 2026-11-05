"""Shared color choices for paper figures."""

from __future__ import annotations

import matplotlib


# Crop exported figures to their content with only a tiny safety margin. A
# zero-width margin can clip rotated labels at the renderer boundary.
matplotlib.rcParams["savefig.bbox"] = "tight"
matplotlib.rcParams["savefig.pad_inches"] = 0.04

# Original main-result palette, used consistently across paper figures.
PALETTE = {
    "red": "#E24A33",
    "blue": "#348ABD",
    # Darkened from #FBC15E to stay in the accessible lightness band
    # (validated: adjacent-pair CVD dE >= 20, normal-vision floor >= 23).
    "yellow": "#DFA32B",
    "purple": "#988ED5",
    "green": "#8EBA42",
    "dark": "#222222",
}

POLICY_COLORS = {
    "edf_best_fit": PALETTE["red"],
    "edf": PALETTE["red"],
    "EDF": PALETTE["red"],
    "srtf_sp1": PALETTE["blue"],
    "SRTF-SP1": PALETTE["blue"],
    "srtf_sp4": PALETTE["yellow"],
    "srtf_sp8": PALETTE["yellow"],
    "srtf_spmax": PALETTE["yellow"],
    "SRTF-SP4": PALETTE["yellow"],
    "SRTF-SPmax": PALETTE["yellow"],
    "fcfs_sp1": PALETTE["purple"],
    "FCFS-SP1": PALETTE["purple"],
    "fcfs_sp4": PALETTE["green"],
    "FCFS-SP4": PALETTE["green"],
    "legacy": PALETTE["green"],
    "Legacy": PALETTE["green"],
    "SJF-SP1": PALETTE["red"],
}

REQUEST_CLASS_COLORS = {
    "S": PALETTE["blue"],
    "M": PALETTE["yellow"],
    "L": PALETTE["green"],
}

COMPONENT_COLORS = {
    "DiT": PALETTE["blue"],
    "VAE": PALETTE["red"],
    "Text encoder": PALETTE["green"],
    "Ideal": PALETTE["dark"],
}

SHAPE_COLORS = [
    PALETTE["blue"],
    PALETTE["yellow"],
    PALETTE["green"],
    PALETTE["red"],
    PALETTE["purple"],
]

SYSTEM_COLORS = {
    "GFC": PALETTE["blue"],
    "NCCL": PALETTE["red"],
    "Simulator": PALETTE["blue"],
    "Real H20": PALETTE["yellow"],
    "latency": PALETTE["red"],
    "throughput": PALETTE["blue"],
}

GRID_COLOR = "#D0D0D0"
REFERENCE_COLOR = "#555555"
TEXT_MUTED = "#555555"
ALL_SERIES_COLOR = PALETTE["dark"]

PHASE_COLORS = ["#F7F7F7", "#EEF3F8", "#F8F2E8", "#EDF5F0", "#F4EFF6"]
GROUP_BACKGROUND_COLORS = {
    "all2all": "#EEF3F8",
    "allgather": "#F8F2E8",
}

SLO_CMAP = "RdYlGn"
