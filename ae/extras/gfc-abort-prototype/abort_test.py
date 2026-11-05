"""Feasibility test for GFC's peer-failure abort path (local, one process per GPU).

MODE=abort    : GFC_ABORT_CHECK>0. The dead rank exits while survivors are spinning in an
                all_gather that includes it; a survivor-side detector thread notices the
                process is gone and calls runtime.declare_dead(). Expect GFCPeerFailure,
                then survivors re-form a group and run correct collectives in-process.
MODE=watchdog : GFC_ABORT_CHECK=0, 3 s watchdog (today's behavior). Expect a kernel trap
                and an unusable CUDA context.
"""
import json, os, statistics, sys, threading, time
from datetime import timedelta

import torch
import torch.distributed as dist
from gfc import GFCPeerFailure, SymmetricCollectiveConfig, SymmetricCollectiveRuntime

rank = int(os.environ["RANK"]); world = int(os.environ["WORLD_SIZE"]); mode = os.environ["MODE"]
dead = world - 1; run_dir = os.environ["RUN_DIR"]
dev = torch.device(f"cuda:{rank}"); torch.cuda.set_device(dev)
dist.init_process_group("nccl", rank=rank, world_size=world, timeout=timedelta(seconds=120))
cfg = SymmetricCollectiveConfig(max_group_size=world, max_collective_bytes=64 << 20,
                                timeout_ms=30_000 if mode == "abort" else 3_000)
rt = SymmetricCollectiveRuntime(cfg, device=dev)
N = 1 << 20  # bf16 elements per rank (2 MiB)
open(f"{run_dir}/pid{rank}", "w").write(str(os.getpid()))


def emit(**kv):
    print("RESULT " + json.dumps({"rank": rank, "mode": mode, "abort_check": rt.abort_check, **kv}), flush=True)


def ag_buffers(g):
    inp = torch.full((N,), float(rank + 1), dtype=torch.bfloat16, device=dev)
    return inp, torch.empty(g.size * N, dtype=torch.bfloat16, device=dev)


def ag_ok(g):
    inp, out = ag_buffers(g); rt.all_gather(inp, out, g); rt.stream.synchronize()
    return all(bool((out[i * N:(i + 1) * N] == float(r + 1)).all()) for i, r in enumerate(g.ranks))


def a2a_ok(g):
    inp = torch.empty(g.size * N, dtype=torch.bfloat16, device=dev)
    for j in range(g.size):
        inp[j * N:(j + 1) * N].fill_(float(rank * 8 + j))
    out = torch.empty_like(inp); rt.all2all(inp, out, g, slice_bytes=N * 2); rt.stream.synchronize()
    me = g.ranks.index(rank)
    return all(bool((out[i * N:(i + 1) * N] == float(r * 8 + me)).all()) for i, r in enumerate(g.ranks))


def median_us(fn, iters):
    lat = []
    for _ in range(iters):
        t0 = time.perf_counter(); fn(); lat.append((time.perf_counter() - t0) * 1e6)
    return round(statistics.median(lat), 1)


# ---- Phase 1: all alive; correctness + latency with/without abort polling ----
g_all = rt.register_group(list(range(world)))
for _ in range(5):
    ag_ok(g_all); a2a_ok(g_all)                       # warm up / JIT
dist.barrier()
bar_us = median_us(lambda: (rt.barrier(g_all), rt.stream.synchronize()), 500)
inp, out = ag_buffers(g_all)
ag_us = median_us(lambda: (rt.all_gather(inp, out, g_all), rt.stream.synchronize()), 200)
emit(phase="healthy", group=list(g_all.ranks), correct=bool(ag_ok(g_all) and a2a_ok(g_all)),
     barrier_median_us=bar_us, all_gather_2MiB_median_us=ag_us)
dist.barrier()  # last point where everyone is known alive

# ---- Phase 2: rank `dead` crashes while the others enter a collective with it ----
# KILL=self    : the victim calls os._exit(1) at a step boundary.
# KILL=external: everyone loops all_gather; the launcher SIGKILLs the victim at a
#                random moment, so it usually dies inside a kernel.
kill_mode = os.environ.get("KILL", "self")
if rank == dead:
    if kill_mode == "self":
        open(f"{run_dir}/death_time", "w").write(repr(time.time()))
        os._exit(1)
    open(f"{run_dir}/looping", "w").write("1")
    inp, out = ag_buffers(g_all)
    while True:                                    # until SIGKILLed by the launcher
        rt.all_gather(inp, out, g_all); rt.stream.synchronize()
victim_pid = int(open(f"{run_dir}/pid{dead}").read())
marks = {}


def detector():  # stands in for the control plane noticing the worker process exit
    while True:
        try:
            state = open(f"/proc/{victim_pid}/stat").read().split()[2]
        except FileNotFoundError:
            state = "gone"
        if state in ("gone", "Z", "X"):
            marks["detected"] = time.time()
            if mode == "abort":
                rt.declare_dead(dead)
            return
        time.sleep(0.001)


threading.Thread(target=detector, daemon=True).start()
inp, out = ag_buffers(g_all)
outcome = {}
loops = 0
try:
    while True:                                     # keep issuing until the victim's death surfaces
        rt.all_gather(inp, out, g_all)
        ev = torch.cuda.Event(); ev.record(rt.stream)
        while not ev.query():
            time.sleep(0.0005)
        rt.stream.synchronize()
        rt.check_abort()
        loops += 1
        if kill_mode == "self":
            outcome["result"] = "completed (unexpected)"; break
except GFCPeerFailure as e:
    outcome["result"] = f"GFCPeerFailure dead_ranks={e.dead_ranks}"
except Exception as e:
    outcome["result"] = f"{type(e).__name__}: {str(e)[:120]}"
t_end = time.time()
outcome["collectives_completed_before_failure"] = loops
death = float(open(f"{run_dir}/death_time").read())
outcome["death_to_detect_ms"] = round((marks.get("detected", float("nan")) - death) * 1e3, 1)
outcome["death_to_error_ms"] = round((t_end - death) * 1e3, 1)
try:
    torch.ones(1, device=dev).add_(1).item(); outcome["cuda_context_usable"] = True
except Exception as e:
    outcome["cuda_context_usable"] = False; outcome["cuda_error"] = f"{type(e).__name__}: {str(e)[:80]}"
emit(phase="peer_crash_mid_collective", **outcome)

# ---- Phase 3: survivors keep going in-process with a regrouped set ----
def file_barrier(tag, ranks, timeout_s=60):  # stands in for the control plane sequencing survivors
    open(f"{run_dir}/{tag}{rank}", "w").write("1")
    t0 = time.time()
    while not all(os.path.exists(f"{run_dir}/{tag}{r}") for r in ranks):
        if time.time() - t0 > timeout_s:
            return False
        time.sleep(0.005)
    return True


if outcome.get("cuda_context_usable"):
    rt.clear_abort()
    survivors = [r for r in range(world) if r != dead]
    if not file_barrier("quiesced", survivors):
        emit(phase="after_regroup", note="a survivor never quiesced (its context is unusable)"); os._exit(0)
    rt.reset_edges([r for r in survivors if r != rank])
    file_barrier("reset", survivors)
    t0 = time.perf_counter(); g_new = rt.register_group(survivors); reg_us = (time.perf_counter() - t0) * 1e6
    ok = all(ag_ok(g_new) and a2a_ok(g_new) for _ in range(20))
    bar2 = median_us(lambda: (rt.barrier(g_new), rt.stream.synchronize()), 200)
    emit(phase="after_regroup", group=survivors, register_us=round(reg_us, 1), correct=bool(ok), barrier_median_us=bar2)
os._exit(0)
