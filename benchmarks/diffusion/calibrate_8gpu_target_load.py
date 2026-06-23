#!/usr/bin/env python3
"""Calibrate 8-GPU target-load per (model, profile) via the mixed-priority sim.

Generates a workload per target-load (num_gpus=8, SP8-augmented cost model) and
runs EDF + static baselines on the sliding 8-rank topology, printing per-policy
overall SLO so a differentiable target-load can be picked. CPU-only.

Each policy runs in its OWN runner process (fresh topology) -- running several
policies in one runner invocation lets EDF's ensure_group() mutate the shared
topology and corrupts later static policies (static_sp4 -> 0 completions). Jobs
run in a thread pool for parallelism.

  .venv/bin/python benchmarks/diffusion/calibrate_8gpu_target_load.py --jobs 16
"""
from __future__ import annotations
import argparse, json, subprocess, os
from concurrent.futures import ThreadPoolExecutor, as_completed

PY = ".venv/bin/python"
GEN = "benchmarks/diffusion/generate_runtime_v2_mixed_priority_workload.py"
RUN = "benchmarks/diffusion/runtime_v2_mixed_priority_runner.py"
ENV = {**os.environ, "PYTHONPATH": os.getcwd() + os.pathsep + os.environ.get("PYTHONPATH", "")}

MODELS = {
    "qwen": dict(cost_dir="cost-model/qwen-image-a100", alpha=(1.5, 2.0, 6.0), deadline_const=1000.0,
                 dur={"short": 600.0, "foreground_burst": 1800.0, "inflight_burst": 1500.0}),
    "wan": dict(cost_dir="cost-model/wan22-ti2v-5b-a100", alpha=(2.0, 2.5, 3.5), deadline_const=5000.0,
                dur={"short": 1200.0, "foreground_burst": 1800.0, "inflight_burst": 1500.0}),
}
# per-profile load grids -- A100, sweep up to where edf_best drops below ~90%
# so we can pick the load giving a loose-SLO ~90-95% edf_best knee.
PROFILE_LOADS = {
    "short":            [0.8, 0.9, 1.0],
    "foreground_burst": [0.5, 0.6, 0.7, 0.8],
    "inflight_burst":   [0.4, 0.45, 0.5, 0.55],
}
POLICIES = [
    ("benchmarks.diffusion.policies.edf_greedy:make_edf_greedy", "edf_greedy"),
    ("benchmarks.diffusion.policies.edf_best_fit:make_edf_best_fit", "edf_best_fit"),
    ("benchmarks.diffusion.policies.static_sp:make_static_sp1", "static_sp1"),
    ("benchmarks.diffusion.policies.static_sp:make_static_sp4", "static_sp4"),
    ("benchmarks.diffusion.policies.static_sp:make_static_sp8", "static_sp8"),
]
NAMES = [n for _, n in POLICIES]


def gen_workload(mk, prof, tl, wl):
    if os.path.exists(wl):
        return
    m = MODELS[mk]; a = m["alpha"]
    cmd = [PY, GEN, "--out", wl, "--cost-model-dir", m["cost_dir"], "--num-gpus", "8",
           "--profile", prof, "--target-load", str(tl),
           "--alpha-s", str(a[0]), "--alpha-m", str(a[1]), "--alpha-l", str(a[2]),
           "--deadline-constant-ms", str(m["deadline_const"]),
           "--duration-s", str(m["dur"][prof]), "--seed", "42"]
    r = subprocess.run(cmd, capture_output=True, text=True, env=ENV)
    if r.returncode != 0:
        raise RuntimeError(f"gen {mk}/{prof}/{tl}: {r.stderr[-400:]}")


def run_policy(mk, wl, spec, name, out):
    m = MODELS[mk]
    cmd = [PY, RUN, "--workload-json", wl, "--cost-model-dir", m["cost_dir"],
           "--policy", spec, "--policy-name", name, "--topology-style", "sliding",
           "--results-out", out]
    r = subprocess.run(cmd, capture_output=True, text=True, env=ENV)
    if r.returncode != 0:
        return None
    res = json.load(open(out))["results"][0]
    bc = res.get("by_class") or {}
    n = sum(bc[k]["n"] for k in bc)
    if not bc or n == 0:
        return None
    return sum(bc[k]["slo_attained"] for k in bc) / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["qwen", "wan"])
    ap.add_argument("--profiles", nargs="+", default=["short", "foreground_burst", "inflight_burst"])
    ap.add_argument("--loads", default=None, help="override all profiles' load grid, comma-separated")
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--workdir", default="out/calib8")
    args = ap.parse_args()
    os.makedirs(args.workdir, exist_ok=True)
    override = [float(x) for x in args.loads.split(",")] if args.loads else None

    # build (mk, prof, tl) points
    points = []
    for mk in args.models:
        for prof in args.profiles:
            for tl in (override or PROFILE_LOADS[prof]):
                points.append((mk, prof, tl))

    # phase 1: generate workloads in parallel
    def _gen(p):
        mk, prof, tl = p
        gen_workload(mk, prof, tl, f"{args.workdir}/{mk}_{prof}_tl{tl}.json")
        return p
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        for f in as_completed([ex.submit(_gen, p) for p in points]):
            f.result()
    print(f"[gen] {len(points)} workloads ready", flush=True)

    # phase 2: run each (point, policy) in its own process, in parallel
    results = {}  # (mk,prof,tl) -> {name: slo}
    jobs = []
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        fut = {}
        for (mk, prof, tl) in points:
            wl = f"{args.workdir}/{mk}_{prof}_tl{tl}.json"
            for spec, name in POLICIES:
                out = f"{args.workdir}/{mk}_{prof}_tl{tl}__{name}.json"
                fut[ex.submit(run_policy, mk, wl, spec, name, out)] = (mk, prof, tl, name)
        done = 0
        for f in as_completed(fut):
            mk, prof, tl, name = fut[f]
            results.setdefault((mk, prof, tl), {})[name] = f.result()
            done += 1
            if done % 10 == 0:
                print(f"[sim] {done}/{len(fut)}", flush=True)

    # phase 3: tabulate
    for mk in args.models:
        for prof in args.profiles:
            print(f"\n###### {mk} / {prof} (8-GPU) ######")
            print(f"{'tl':>5}  " + " ".join(f"{n:>13}" for n in NAMES))
            for tl in (override or PROFILE_LOADS[prof]):
                c = results.get((mk, prof, tl), {})
                cells = " ".join(
                    (f"{c[n]*100:>12.1f}%" if c.get(n) is not None else f"{'n/a':>13}") for n in NAMES)
                print(f"{tl:>5}  {cells}")


if __name__ == "__main__":
    main()
