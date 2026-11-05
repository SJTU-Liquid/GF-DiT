"""Re-run every simulator-backed claim of the paper and check it against the paper.

Standard library only (runs on any CPU with Python >= 3.10). From the artifact root:

    python3 ae/sim/run_claims.py            # all claims (~10-20 min on 8 cores)
    python3 ae/sim/run_claims.py --quick    # S1 + S2 only (< 1 min)

Claims (see ae/README.md for the mapping to the paper):
  S1  Sec. 6.9   TetriServe vs. EDF in simulation, H20 Wan2.2 short / fg.
  S2  Fig. 10    Simulator bars, H20 Wan2.2 foreground burst.
  S3  Sec. 6.6   Scalability: 8 vs. 64 GPUs, GFC vs. world-synchronized new_group.
  S4  Sec. 6.4   (supporting) SRTF-SP1 tie-break: released arrival order vs. the
                 earlier random tie-break used for Fig. 6.
Results go to ae/sim/out/; a summary table is printed and written to
ae/sim/out/summary.md, and the exit code is non-zero if any check fails.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OUT = HERE / "out"
SHIM = HERE / "sim_shim.py"
RUNNER = "benchmarks.diffusion.runtime_v2_mixed_priority_runner"
COST = HERE / "cost-model" / "wan22-ti2v-5b-h20"
WL_SHORT = HERE / "workloads" / "h20bat_short_tl1.20_a2.0_2.5_3.5_seed42.json"
WL_FG = HERE / "workloads" / "h20bat_foreground_burst_tl1.10_a2.0_2.5_3.5_seed42.json"

P = "benchmarks.diffusion.policies"
EDF = f"{P}.edf_best_fit:make_edf_best_fit"
TETRI = f"{P}.tetriserve:make_tetriserve_round"
SRTF = f"{P}.static_sp:make_srtf"
FCFS = f"{P}.static_sp:make_fcfs"
SRTF_LEGACY = "ae_policies:make_srtf_legacy_tiebreak"
SP1 = "sp1_0,sp1_1,sp1_2,sp1_3"


# --------------------------------------------------------------------------- helpers
def randomize_ids(src: Path, seed: int) -> Path:
    """Copy of a workload with md5-randomized request ids (as real serving's uuids)."""
    dst = OUT / "workloads" / f"{src.stem}__randid{seed}.json"
    if not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        d = json.loads(src.read_text())
        for r in d["requests"]:
            r["request_id"] = "vg_" + hashlib.md5(f"{seed}:{r['request_id']}".encode()).hexdigest()
        dst.write_text(json.dumps(d))
    return dst


def scale_topology(n: int) -> str:
    groups = [{"group_id": f"sp1_{i}", "ranks": [i], "ulysses_degree": 1, "tp": 1} for i in range(n)]
    big = min(n, 8)
    groups += [{"group_id": f"spbig_{k}", "ranks": list(range(big * k, big * k + big)),
                "ulysses_degree": big, "tp": 1} for k in range(n // big)]
    return json.dumps(groups)


def run(job: dict) -> dict:
    out = Path(job["out"])
    out.parent.mkdir(parents=True, exist_ok=True)
    if not out.exists():
        cmd = [sys.executable, "-I", str(SHIM), job.get("module", RUNNER),
               "--workload-json", str(job["wl"]), "--cost-model-dir", str(COST),
               "--policy", job["policy"], "--policy-name", job["name"],
               "--world-size", str(job["world"]), "--reshard-ms", "3.0", "--results-out", str(out)]
        cmd += ["--topology-json", job["topology"]] if job.get("topology") else ["--topology-style", "sliding"]
        if job.get("allowed"):
            cmd += ["--allowed-groups", job["allowed"]]
        cmd += job.get("extra_args", [])
        env = {**os.environ, **job.get("env", {})}
        r = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True)
        Path(str(out) + ".log").write_text(r.stdout + r.stderr)
        if r.returncode != 0:
            return {**job, "error": r.stderr[-400:]}
    d = json.loads(out.read_text())
    res = d["results"][0]
    bc = res["by_class"]
    n = res["overall"]["total_requests"]
    return {**job, "slo": 100.0 * sum(v["slo_attained"] for v in bc.values()) / n,
            "mean_s": res["overall"]["avg_latency_ms"] / 1000.0, "extra": d.get("bd_extra", {})}


# --------------------------------------------------------------------------- claims
def jobs_s1():
    js = []
    for tag, wl in (("short", WL_SHORT), ("fg", WL_FG)):
        js.append(dict(claim="S1", key=f"tetriserve/{tag}", wl=wl, policy=TETRI, name="tetriserve",
                       world=4, extra_args=["--round-ms", "0"], out=OUT / "S1" / f"tetriserve_{tag}.json"))
        js.append(dict(claim="S1", key=f"edf/{tag}", wl=wl, policy=EDF, name="edf_best_fit",
                       world=4, out=OUT / "S1" / f"edf_{tag}.json"))
    return js


def jobs_s2():
    fg_rand = randomize_ids(WL_FG, 0)
    js = [dict(claim="S2", key="EDF", wl=WL_FG, policy=EDF, name="edf_best_fit", world=4, out=OUT / "S2" / "edf.json"),
          dict(claim="S2", key="FCFS-SP1", wl=WL_FG, policy=FCFS, name="fcfs_sp1", world=4, allowed=SP1,
               out=OUT / "S2" / "fcfs_sp1.json"),
          dict(claim="S2", key="FCFS-SP4", wl=WL_FG, policy=FCFS, name="fcfs_sp4", world=4, allowed="sp4",
               out=OUT / "S2" / "fcfs_sp4.json")]
    for key, allowed in (("SRTF-SP1", SP1), ("SRTF-SPmax", "sp4")):
        js.append(dict(claim="S2", key=key, wl=fg_rand, policy=SRTF_LEGACY, name=key, world=4, allowed=allowed,
                       out=OUT / "S2" / f"{key}_legacy_tiebreak.json"))
        js.append(dict(claim="S2-info", key=f"{key} (released tie-break)", wl=fg_rand, policy=SRTF, name=key,
                       world=4, allowed=allowed, out=OUT / "S2" / f"{key}_released.json"))
    return js


def jobs_s3():
    js = []
    for n in (8, 64):
        for seed in (42, 43, 44):
            wl = HERE / "workloads" / "scale" / f"short_n{n}_tl1.2_s{seed}.json"
            for mode, label in (("none", "gfc"), ("world", "nccl_world")):
                js.append(dict(claim="S3", key=f"n{n}/{label}", seed=seed, module="bd_sim", wl=wl, policy=EDF,
                               name=f"edf_{label}", world=n, topology=scale_topology(n),
                               env={"BD_SETUP_MODE": mode, "EDF_GREEDY_SP_SIZES": ",".join(
                                   str(s) for s in (1, 2, 4, 8) if s <= n), "BD_FREE_ALIGNED_SP": str(min(n, 8))},
                               out=OUT / "S3" / f"n{n}_s{seed}_{label}.json"))
    return js


def jobs_s4():
    js = [dict(claim="S4", key="released", wl=WL_SHORT, policy=SRTF, name="srtf_sp1", world=4, allowed=SP1,
               out=OUT / "S4" / "srtf_sp1_released.json")]
    for seed in range(6):
        js.append(dict(claim="S4", key="legacy", seed=seed, wl=randomize_ids(WL_SHORT, seed), policy=SRTF_LEGACY,
                       name="srtf_sp1", world=4, allowed=SP1, out=OUT / "S4" / f"srtf_sp1_legacy_rand{seed}.json"))
    return js


# --------------------------------------------------------------------------- checks
def close(x, want, tol):
    return abs(x - want) <= tol


def summarize(results: list[dict]) -> tuple[list[str], bool]:
    by = {}
    for r in results:
        by.setdefault((r["claim"], r["key"]), []).append(r)
    lines, ok_all = [], True

    def row(claim, what, got, want, ok):
        nonlocal ok_all
        ok_all &= ok
        lines.append(f"| {claim} | {what} | {got} | {want} | {'PASS' if ok else 'FAIL'} |")

    lines += ["| claim | quantity | artifact | paper | check |", "|---|---|---|---|---|"]
    for (c, k), rs in sorted(by.items()):
        if any("error" in r for r in rs):
            row(c, k, "ERROR", "-", False)
    # S1
    want = {"tetriserve/short": 59.0, "tetriserve/fg": 52.3, "edf/short": 91.8, "edf/fg": 89.2}
    for k, w in want.items():
        rs = by.get(("S1", k))
        if rs and "slo" in rs[0]:
            row("S1 (Sec. 6.9)", f"{k} SLO %", f"{rs[0]['slo']:.1f}", f"{w:.1f}", close(rs[0]["slo"], w, 0.05))
    # S2
    want = {"EDF": 89.2, "SRTF-SP1": 81.5, "SRTF-SPmax": 55.4, "FCFS-SP1": 49.2, "FCFS-SP4": 21.5}
    for k, w in want.items():
        rs = by.get(("S2", k))
        if rs and "slo" in rs[0]:
            row("S2 (Fig. 10)", f"{k} sim SLO %", f"{rs[0]['slo']:.1f}", f"{w:.1f}", close(rs[0]["slo"], w, 0.06))
    for (c, k), rs in sorted(by.items()):
        if c == "S2-info" and "slo" in rs[0]:
            lines.append(f"| S2 (info) | {k} sim SLO % | {rs[0]['slo']:.1f} | - | info |")
    # S3
    def mean(key, f):
        rs = by.get(("S3", key), [])
        vals = [f(r) for r in rs if "mean_s" in r]
        return statistics.mean(vals) if len(vals) == 3 else None
    lat = {k: mean(k, lambda r: r["mean_s"]) for k in ("n8/gfc", "n8/nccl_world", "n64/gfc", "n64/nccl_world")}
    rcs = {n: mean(f"n{n}/gfc", lambda r: r["extra"].get("reconfigs_per_s", float("nan"))) for n in (8, 64)}
    comm64 = mean("n64/gfc", lambda r: r["extra"].get("max_comms_per_gpu_if_cached", float("nan")))
    mem64 = mean("n64/gfc", lambda r: r["extra"].get("max_comm_mem_gb_per_gpu_if_cached", float("nan")))
    if None not in rcs.values():
        row("S3 (Sec. 6.6)", "reconfigs/s at 8 -> 64 GPUs", f"{rcs[8]:.2f} -> {rcs[64]:.2f}", "0.09 -> 0.96",
            close(rcs[8], 0.09, 0.006) and close(rcs[64], 0.96, 0.006))
    for n, w in ((8, 14), (64, 30)):
        g, wv = lat.get(f"n{n}/gfc"), lat.get(f"n{n}/nccl_world")
        if g and wv:
            inc = 100 * (wv / g - 1)
            row("S3 (Sec. 6.6)", f"world-sync new_group latency increase, {n} GPUs",
                f"+{inc:.1f}% ({g:.1f}s -> {wv:.1f}s)", f"+{w}%", close(inc, w, 0.5))
    if comm64 is not None:
        # The simulator counts one communicator per rank set; vLLM-Omni needs two
        # (SP group + Ulysses subgroup), hence the factor of 2 in the paper.
        row("S3 (Sec. 6.6)", "communicators per GPU if cached, 64 GPUs (x2)",
            f"{2 * comm64:.0f} ({2 * mem64:.1f} GB)", "70 (~36 GB)", close(2 * comm64, 70, 0.5) and close(2 * mem64, 36, 0.6))
    # S4
    rel = by.get(("S4", "released"), [{}])[0].get("slo")
    leg = [r["slo"] for r in by.get(("S4", "legacy"), []) if "slo" in r]
    if rel is not None and leg:
        lines.append(f"| S4 (Sec. 6.4, supporting) | SRTF-SP1 SLO %, released vs. legacy random tie-break "
                     f"(mean of {len(leg)} id seeds) | {rel:.1f} vs. {statistics.mean(leg):.1f} | "
                     f"real: 70.5 vs. 82.0 | info |")
    return lines, ok_all


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="only S1 and S2 (< 1 minute)")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--fresh", action="store_true", help="ignore cached results in ae/sim/out")
    args = ap.parse_args()
    if args.fresh and OUT.exists():
        for p in OUT.rglob("*.json"):
            p.unlink()
    jobs = jobs_s1() + jobs_s2() + ([] if args.quick else jobs_s3() + jobs_s4())
    print(f"running {len(jobs)} simulations with {args.jobs} workers ...", flush=True)
    with ThreadPoolExecutor(args.jobs) as ex:
        results = list(ex.map(run, jobs))
    lines, ok = summarize(results)
    text = "\n".join(lines)
    print(text)
    (OUT / "summary.md").write_text(text + "\n")
    print(f"\nwrote {OUT / 'summary.md'}; overall: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
