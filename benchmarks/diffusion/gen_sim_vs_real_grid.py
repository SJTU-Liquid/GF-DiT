#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Regenerate aligned simulator results for the real h20-4card / A100-8card
policy batteries, and emit a sim-vs-real SLO comparison (paper口径).

For each campaign we feed the simulator (runtime_v2_mixed_priority_runner) the
*exact same workload JSON* the real serving run replayed -- so arrivals,
deadlines and target-load are identical -- on the hardware-native cost model.
Each (workload x policy) runs in its OWN runner process (EDF's ensure_group()
mutates the shared topology, so co-running policies would corrupt each other).

Policy -> sim mapping:
  * EDF (edf_best_fit / edf_greedy): full `sliding` topology, idealized
    reshard=3ms (the established sim baseline; the real reshard ~tens-100ms is
    exactly the sim<->real gap the paper reports).
  * static srtf_spK / fcfs_spK: SRTF/FCFS ordering restricted via --allowed-groups
    to the SP-K lanes, mirroring how the real sweep pins fcfs/srtf to a groups-json.
  * bare `srtf`: SRTF over the full sliding topology (elastic, any SP width).

Real SLO is read from `per_class[k].slo_attainment_rate` (serving schema); sim
SLO from `by_class[k].slo_attainment_ratio` (simulator schema).

  PYTHONPATH=. .venv/bin/python benchmarks/diffusion/gen_sim_vs_real_grid.py \
      --campaign h20 a100 --jobs 12 --out-dir out/sim_vs_real
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PY = ".venv/bin/python"
RUN = "benchmarks/diffusion/runtime_v2_mixed_priority_runner.py"
ENV = {**os.environ, "PYTHONPATH": os.getcwd() + os.pathsep + os.environ.get("PYTHONPATH", "")}

EDF = "benchmarks.diffusion.policies.edf_best_fit:make_edf_best_fit"
EDG = "benchmarks.diffusion.policies.edf_greedy:make_edf_greedy"
SRTF = "benchmarks.diffusion.policies.static_sp:make_srtf"
FCFS = "benchmarks.diffusion.policies.static_sp:make_fcfs"

CLASSES = ["S", "M", "L"]


def _sp1(w):  # all SP1 lanes
    return ",".join(f"sp1_{i}" for i in range(w))


# world=4 sliding: sp2_0=[0,1], sp2_2=[2,3] are the 2 non-overlapping SP2 lanes;
# sp4 is the single full group. world=8 analogues below.
H20_POLICIES = {
    "edf_best_fit": (EDF, None),
    "edf_greedy": (EDG, None),
    "srtf_sp1": (SRTF, _sp1(4)),
    "srtf_sp2": (SRTF, "sp2_0,sp2_2"),
    "srtf_sp4": (SRTF, "sp4"),
    "fcfs_sp1": (FCFS, _sp1(4)),
    "fcfs_sp2": (FCFS, "sp2_0,sp2_2"),
    "fcfs_sp4": (FCFS, "sp4"),
}

A100_POLICIES = {
    "edf_best_fit": (EDF, None),
    "edf_greedy": (EDG, None),
    "srtf_sp1": (SRTF, _sp1(8)),
    "srtf_sp8": (SRTF, "sp8"),   # SRTF pinned to the single SP8 group
    "fcfs_sp1": (FCFS, _sp1(8)),
    "legacy": (FCFS, "sp8"),     # SP8 FIFO baseline == FCFS on the SP8 group
}

# real result filenames are inconsistent: most are sweep_<policy>_result.json,
# but the real "srtf_sp8" row is the bare srtf file, and legacy has no sweep_ prefix.
REAL_BASENAMES = {
    "srtf_sp8": ["sweep_srtf_result.json", "sweep_srtf_sp8_result.json"],
    "legacy": ["legacy_sp8_result.json", "sweep_legacy_result.json"],
}

CAMPAIGNS = {
    "h20": {
        "world_size": 4,
        "policies": H20_POLICIES,
        "real_roots": ["out/_h20real/out/h20battery"],
        "workloads": [
            {"row": "WAN short",   "wl": "out/stress/h20bat_short_tl1.20_a2.0_2.5_3.5_seed42.json",
             "cost": "cost-model/wan22-ti2v-5b-h20"},
            {"row": "WAN fg",      "wl": "out/stress/h20bat_foreground_burst_tl1.10_a2.0_2.5_3.5_seed42.json",
             "cost": "cost-model/wan22-ti2v-5b-h20"},
            {"row": "WAN inflight","wl": "out/stress/h20bat_inflight_burst_tl0.90_a2.0_2.5_3.5_seed42.json",
             "cost": "cost-model/wan22-ti2v-5b-h20"},
            {"row": "WAN trident", "wl": "out/stress/h20bat_tridentserve_tl1.09_a2.0_2.5_3.5_seed42.json",
             "cost": "cost-model/wan22-ti2v-5b-h20"},
        ],
    },
    "a100": {
        "world_size": 8,
        "policies": A100_POLICIES,
        "real_roots": [
            "out/a100_session2/out",
            "out/wan_a100_bundle/out",
        ],
        "workloads": [
            {"row": "QWEN short", "wl": "out/a100_session2/out/stress/qwenbat8a100_short_tl1.0_origalpha_seed42.json",
             "cost": "cost-model/qwen-image-a100"},
            {"row": "QWEN fg",    "wl": "out/a100_session2/out/stress/qwenbat8a100_fg_tl0.7_origalpha_seed42.json",
             "cost": "cost-model/qwen-image-a100"},
            {"row": "WAN short",  "wl": "out/wan_a100_bundle/out/stress/wanbat8a100_short_tl1.05_seed42.json",
             "cost": "cost-model/wan22-ti2v-5b-a100",
             "tokens": ["wanbat8a100_short_tl1.05_seed42", "wan_legacy/short"]},
            {"row": "WAN fg",     "wl": "out/wan_a100_bundle/out/stress/wanbat8a100_foreground_burst_tl0.9_seed42.json",
             "cost": "cost-model/wan22-ti2v-5b-a100",
             "tokens": ["wanbat8a100_foreground_burst_tl0.9_seed42", "wan_legacy/fg"]},
        ],
    },
}


def run_sim(world, wl, cost, spec, allowed, name, out):
    cmd = [PY, RUN, "--workload-json", wl, "--cost-model-dir", cost,
           "--policy", spec, "--policy-name", name,
           "--topology-style", "sliding", "--world-size", str(world),
           "--reshard-ms", "3.0", "--results-out", out]
    if allowed:
        cmd += ["--allowed-groups", allowed]
    r = subprocess.run(cmd, capture_output=True, text=True, env=ENV)
    if r.returncode != 0:
        return {"ok": False, "err": r.stderr[-300:]}
    return {"ok": True}


def _inner(d):
    return d["results"][0] if isinstance(d, dict) and "results" in d else d


def slo_from_sim(path):
    """sim by_class -> {class: pct, 'overall': pct}."""
    r = _inner(json.load(open(path)))
    bc = r.get("by_class") or {}
    out, att, tot = {}, 0, 0
    for k in CLASSES:
        if k in bc and bc[k].get("n"):
            out[k] = 100.0 * bc[k]["slo_attained"] / bc[k]["n"]
            att += bc[k]["slo_attained"]; tot += bc[k]["n"]
    out["overall"] = 100.0 * att / tot if tot else None
    return out


def slo_from_real(path):
    """real per_class -> {class: pct, 'overall': pct} using slo_met/slo_defined."""
    r = _inner(json.load(open(path)))
    pc = r.get("per_class") or {}
    out, met, defn = {}, 0, 0
    for k in CLASSES:
        if k in pc and pc[k].get("slo_defined"):
            out[k] = 100.0 * pc[k]["slo_met"] / pc[k]["slo_defined"]
            met += pc[k]["slo_met"]
            defn += pc[k]["slo_defined"]
    out["overall"] = 100.0 * met / defn if defn else None
    return out


def find_real(real_roots, wl, policy):
    """Locate the real result JSON for (workload, policy).

    Handles inconsistent real filenames (REAL_BASENAMES) and inconsistent
    workload-dir naming (wl['tokens']: the wan legacy dir is short/fg, not the
    full workload stem)."""
    tokens = wl.get("tokens") or [Path(wl["wl"]).stem]
    basenames = REAL_BASENAMES.get(policy, [f"sweep_{policy}_result.json"])
    for root in real_roots:
        for base in basenames:
            for hit in glob.glob(f"{root}/**/{base}", recursive=True):
                if any(tok in hit for tok in tokens):
                    return hit
    return None


def fmt(v):
    return f"{v:5.1f}" if isinstance(v, (int, float)) else "  -- "


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--campaign", nargs="+", default=["h20", "a100"], choices=["h20", "a100"])
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--out-dir", default="out/sim_vs_real")
    args = ap.parse_args()

    for camp_name in args.campaign:
        camp = CAMPAIGNS[camp_name]
        world = camp["world_size"]
        sim_dir = Path(args.out_dir) / camp_name / "sim"
        sim_dir.mkdir(parents=True, exist_ok=True)

        # launch all sim cells in parallel
        tasks = []
        for wl in camp["workloads"]:
            for pname, (spec, allowed) in camp["policies"].items():
                out = str(sim_dir / f"{Path(wl['wl']).stem}__{pname}.json")
                tasks.append((wl, pname, spec, allowed, out))
        print(f"[{camp_name}] launching {len(tasks)} sim runs (world={world}) ...", flush=True)
        with ThreadPoolExecutor(max_workers=args.jobs) as ex:
            fut = {ex.submit(run_sim, world, t[0]["wl"], t[0]["cost"], t[2], t[3], t[1], t[4]): t
                   for t in tasks}
            done = 0
            for f in as_completed(fut):
                t = fut[f]; res = f.result(); done += 1
                if not res["ok"]:
                    print(f"  ! FAIL {t[0]['row']}/{t[1]}: {res['err']}", flush=True)
                if done % 8 == 0:
                    print(f"  ... {done}/{len(tasks)}", flush=True)

        # collect grid + sim-vs-real
        grid = []
        for wl in camp["workloads"]:
            for pname in camp["policies"]:
                sim_path = sim_dir / f"{Path(wl['wl']).stem}__{pname}.json"
                sim = slo_from_sim(sim_path) if sim_path.exists() else {}
                real_path = find_real(camp["real_roots"], wl, pname)
                real = slo_from_real(real_path) if real_path else {}
                grid.append({"row": wl["row"], "workload": wl["wl"], "policy": pname,
                             "sim": sim, "real": real,
                             "real_path": real_path})

        out_root = Path(args.out_dir) / camp_name
        (out_root / "grid.json").write_text(json.dumps(grid, indent=2))

        # markdown sim-vs-real table
        lines = [f"# Sim vs Real — {camp_name} ({'4×H20' if camp_name=='h20' else '8×A100'})",
                 "",
                 "SLO attainment %, `sim / real`. sim = idealized reshard=3ms baseline. "
                 "`--` = real cell not run.", ""]
        for wl in camp["workloads"]:
            lines += [f"## {wl['row']}  (`{Path(wl['wl']).name}`)", "",
                      "| policy | S sim/real | M sim/real | L sim/real | overall sim/real |",
                      "|---|---|---|---|---|"]
            for g in [x for x in grid if x["workload"] == wl["wl"]]:
                cells = []
                for k in ["S", "M", "L", "overall"]:
                    cells.append(f"{fmt(g['sim'].get(k))} / {fmt(g['real'].get(k))}")
                lines.append(f"| {g['policy']} | " + " | ".join(cells) + " |")
            lines.append("")
        (out_root / "sim_vs_real.md").write_text("\n".join(lines))

        # flat CSV
        with (out_root / "sim_vs_real.csv").open("w") as f:
            f.write("row,workload,policy,scope,sim_slo,real_slo\n")
            for g in grid:
                for k in ["S", "M", "L", "overall"]:
                    s = g["sim"].get(k); r = g["real"].get(k)
                    f.write(f"{g['row']},{Path(g['workload']).name},{g['policy']},{k},"
                            f"{'' if s is None else f'{s:.2f}'},{'' if r is None else f'{r:.2f}'}\n")

        print(f"[{camp_name}] wrote {out_root/'grid.json'}, sim_vs_real.md, sim_vs_real.csv", flush=True)
        print("\n" + (out_root / "sim_vs_real.md").read_text())


if __name__ == "__main__":
    main()
