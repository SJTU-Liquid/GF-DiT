#!/usr/bin/env python3
"""Headline numbers of the abstract / Sec. 6.3 from the Fig. 6 data.

EDF vs. Legacy over all main-result settings (platform x model x workload):
throughput gain, mean-latency reduction and SLO-violation-rate reduction, each
the maximum over the settings.  Reads figs/main_result_combined.csv (written by
plot_main_result_combined.py).  Run from ae/figures.
"""
from __future__ import annotations

import csv
from pathlib import Path

CSV = Path(__file__).resolve().parents[1] / "figs" / "main_result_combined.csv"


def main() -> None:
    rows = list(csv.DictReader(CSV.open()))
    by = {(r["gpu"], r["model"], r["workload"], r["policy_label"]): r for r in rows}
    best = {"thr": (0.0, None), "lat": (0.0, None), "vio": (0.0, None)}
    for (gpu, model, wl, pol), e in by.items():
        if pol != "EDF":
            continue
        lg = by[(gpu, model, wl, "Legacy")]
        key = f"{gpu} {model} {wl}"
        cand = {"thr": float(e["throughput_qps"]) / float(lg["throughput_qps"]),
                "lat": 1 - float(e["latency_mean_s"]) / float(lg["latency_mean_s"]),
                "vio": 1 - (1 - float(e["slo_attainment"])) / (1 - float(lg["slo_attainment"]))}
        for k, v in cand.items():
            if v > best[k][0]:
                best[k] = (v, key)
    lines = [f"throughput gain        : {best['thr'][0]:.2f}x   ({best['thr'][1]}; paper: 6.01x)",
             f"mean-latency reduction : {100 * best['lat'][0]:.1f}%   ({best['lat'][1]}; paper: 95.2%)",
             f"SLO-violation reduction: {100 * best['vio'][0]:.1f}%   ({best['vio'][1]}; paper: 89.6%)"]
    out = CSV.parent / "headline_numbers.md"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
