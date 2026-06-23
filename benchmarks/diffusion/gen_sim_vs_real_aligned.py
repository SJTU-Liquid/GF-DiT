#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Aligned sim-vs-real regeneration: identical to gen_sim_vs_real_grid.py in
EVERY respect (same policies, cost models, topology, reshard, real lookup) --
the ONLY change is that the simulator's workload request_ids are md5-randomized.

Why: SRTFSchedulerPolicy breaks ties (equal remaining_work, e.g. all fresh S)
by `request_id` string. The stock workloads use ORDERED ids (S_000000,...) ->
the tie-break degrades to FIFO (fair). The REAL serving assigns RANDOM uuids
(video_gen_<hex>) -> the tie-break is effectively random -> some requests are
starved (heavy latency tail). The ordered-id sim therefore does NOT model real.
Randomizing the sim ids restores the random tie-break and aligns the two sides.

This reuses gen_sim_vs_real_grid's helpers verbatim (so the sim config is
provably the same; cf. the 3 alignment checks). Seed 0 is the primary table;
extra seeds give a mean +/- std robustness check on the SRTF rows (the only
rows the tie-break can move).

  PYTHONPATH=. .venv/bin/python benchmarks/diffusion/gen_sim_vs_real_aligned.py \
      --campaign h20 --seeds 6 --jobs 12 --out-dir out/sim_vs_real_aligned
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from benchmarks.diffusion.gen_sim_vs_real_grid import (
    CAMPAIGNS,
    CLASSES,
    find_real,
    fmt,
    run_sim,
    slo_from_real,
    slo_from_sim,
)


def randomize_ids(wl_path: str, seed: int, out_path: Path) -> str:
    """Write a copy of the workload with md5-randomized request_ids (deterministic
    per seed). EVERYTHING else (arrival_ms, deadline_ms, num_steps, request_class,
    sizes) is byte-identical -- only the id changes, so only the SRTF id tie-break
    is affected, not cost/execution/classification."""
    wl = json.loads(Path(wl_path).read_text())
    for r in wl["requests"]:
        r["request_id"] = "vg_" + hashlib.md5(f"{seed}:{r['request_id']}".encode()).hexdigest()
    out_path.write_text(json.dumps(wl))
    return str(out_path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--campaign", nargs="+", default=["h20"], choices=["h20", "a100"])
    ap.add_argument("--seeds", type=int, default=6, help="number of random-id seeds (>=1)")
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--out-dir", default="out/sim_vs_real_aligned")
    args = ap.parse_args()

    for camp_name in args.campaign:
        camp = CAMPAIGNS[camp_name]
        world = camp["world_size"]
        out_root = Path(args.out_dir) / camp_name
        wl_dir = out_root / "randwl"
        wl_dir.mkdir(parents=True, exist_ok=True)

        # --- build every (seed, workload) randomized copy up front ---
        rand_wl = {}  # (seed, orig_wl_path) -> randomized path
        for seed in range(args.seeds):
            for wl in camp["workloads"]:
                stem = Path(wl["wl"]).stem
                rand_wl[(seed, wl["wl"])] = randomize_ids(
                    wl["wl"], seed, wl_dir / f"{stem}__seed{seed}.json"
                )

        # --- launch sim cells: seed 0 = ALL policies; seeds>=1 = SRTF rows only ---
        sim_dir = out_root / "sim"
        sim_dir.mkdir(parents=True, exist_ok=True)
        tasks = []
        for seed in range(args.seeds):
            for wl in camp["workloads"]:
                for pname, (spec, allowed) in camp["policies"].items():
                    if seed > 0 and not pname.startswith("srtf"):
                        continue
                    out = sim_dir / f"{Path(wl['wl']).stem}__{pname}__seed{seed}.json"
                    tasks.append((seed, wl, pname, spec, allowed, str(out)))
        print(f"[{camp_name}] launching {len(tasks)} aligned sim runs "
              f"({args.seeds} seeds, world={world}) ...", flush=True)
        with ThreadPoolExecutor(max_workers=args.jobs) as ex:
            fut = {
                ex.submit(run_sim, world, rand_wl[(t[0], t[1]["wl"])], t[1]["cost"],
                          t[3], t[4], t[2], t[5]): t
                for t in tasks
            }
            done = 0
            for f in as_completed(fut):
                t = fut[f]; res = f.result(); done += 1
                if not res["ok"]:
                    print(f"  ! FAIL seed{t[0]} {t[1]['row']}/{t[2]}: {res['err']}", flush=True)
                if done % 16 == 0:
                    print(f"  ... {done}/{len(tasks)}", flush=True)

        # --- primary aligned grid (seed 0) ---
        grid = []
        for wl in camp["workloads"]:
            for pname in camp["policies"]:
                sp = sim_dir / f"{Path(wl['wl']).stem}__{pname}__seed0.json"
                sim = slo_from_sim(sp) if sp.exists() else {}
                rp = find_real(camp["real_roots"], wl, pname)
                real = slo_from_real(rp) if rp else {}
                # seed spread (overall) for SRTF rows
                spread = None
                if pname.startswith("srtf"):
                    ov = []
                    for seed in range(args.seeds):
                        q = sim_dir / f"{Path(wl['wl']).stem}__{pname}__seed{seed}.json"
                        if q.exists():
                            v = slo_from_sim(q).get("overall")
                            if v is not None:
                                ov.append(v)
                    if len(ov) > 1:
                        spread = (statistics.mean(ov), statistics.pstdev(ov), min(ov), max(ov), len(ov))
                grid.append({"row": wl["row"], "workload": wl["wl"], "policy": pname,
                             "sim": sim, "real": real, "real_path": rp, "seed_spread": spread})
        (out_root / "grid.json").write_text(json.dumps(grid, indent=2))

        # --- markdown ---
        lines = [f"# Sim vs Real — {camp_name} (ALIGNED: md5-random sim ids)", "",
                 "SLO attainment %, `sim / real`. sim = idealized reshard=3ms, "
                 "**request_ids md5-randomized (seed 0)** so the SRTF id tie-break "
                 "matches real's random uuids. `srtf*` rows also show seed mean±std.", ""]
        for wl in camp["workloads"]:
            lines += [f"## {wl['row']}  (`{Path(wl['wl']).name}`)", "",
                      "| policy | S sim/real | M sim/real | L sim/real | overall sim/real | srtf seed mean±std |",
                      "|---|---|---|---|---|---|"]
            for g in [x for x in grid if x["workload"] == wl["wl"]]:
                cells = [f"{fmt(g['sim'].get(k))} / {fmt(g['real'].get(k))}" for k in ["S", "M", "L", "overall"]]
                sp = g["seed_spread"]
                spread_s = (f"{sp[0]:.1f}±{sp[1]:.1f} (n={sp[4]}, {sp[2]:.1f}–{sp[3]:.1f})"
                            if sp else "")
                lines.append(f"| {g['policy']} | " + " | ".join(cells) + f" | {spread_s} |")
            lines.append("")
        (out_root / "sim_vs_real.md").write_text("\n".join(lines))

        # --- flat csv ---
        with (out_root / "sim_vs_real.csv").open("w") as f:
            f.write("row,workload,policy,scope,sim_slo,real_slo\n")
            for g in grid:
                for k in ["S", "M", "L", "overall"]:
                    s = g["sim"].get(k); r = g["real"].get(k)
                    f.write(f"{g['row']},{Path(g['workload']).name},{g['policy']},{k},"
                            f"{'' if s is None else f'{s:.2f}'},{'' if r is None else f'{r:.2f}'}\n")

        print(f"[{camp_name}] wrote {out_root/'sim_vs_real.md'}")
        print("\n" + (out_root / "sim_vs_real.md").read_text())


if __name__ == "__main__":
    main()
