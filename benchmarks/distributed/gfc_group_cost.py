#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Benchmark the cost of creating a GFC (GroupFree-Collective) group.

Counterpart to ``nccl_subgroup_cost.py``. The thesis: a GFC group is NOT an
NCCL-style communicator -- ``register_group(ranks)`` is essentially a dict
insert + a per-group device rank list allocated on demand (a tiny H2D copy),
with NO per-group communicator init and NO per-group memory. The symmetric-
memory region is allocated ONCE in the runtime constructor (a collective
rendezvous), shared by every group, independent of group count.

Launch on N GPUs (auto world size):

    torchrun --standalone --nproc_per_node=8 \
        benchmarks/distributed/gfc_group_cost.py

Measures:
  * runtime init (one-time)  : SymmetricCollectiveRuntime() wall time + the
                               symm-region memory it allocates (shared by ALL
                               groups; the GFC analogue of NCCL's per-comm cost,
                               but paid once).
  * register_group (per size): host wall time of register_group(range(s)) +
                               first/warm all_gather on it + free-mem delta.
  * churn                    : K register_group() calls back-to-back -> per-call
                               time + CUMULATIVE free-mem drop (expected ~flat:
                               each group's rank list is a tiny uint32[size]).

Compare against nccl_subgroup_cost.py: NCCL new_group ~0.5ms + first collective
0.2-1.0s (lazy ncclCommInitRank) + ~520MB/communicator accumulating linearly.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch
import torch.distributed as dist

CLASSES = None


def _sync(dev):
    torch.cuda.synchronize(dev)


def _free_mb(dev):
    free, _ = torch.cuda.mem_get_info(dev)
    return free / (1024 ** 2)


def _all_gather(runtime, desc, dev, nbytes=65536):
    n = nbytes // 4
    inp = torch.randn(n, device=dev)
    out = torch.empty(n * desc.size, device=dev)
    _sync(dev)
    t0 = time.perf_counter()
    runtime.all_gather(inp, out, group=desc)
    _sync(dev)
    return (time.perf_counter() - t0) * 1e3


def _gather(obj):
    out = [None] * dist.get_world_size()
    dist.all_gather_object(out, obj)
    return out


def _mx(xs):
    return max(xs) if xs else float("nan")


def _mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def default_sizes(world):
    s, p = [], 2
    while p <= world:
        s.append(p)
        p *= 2
    if world >= 2 and world not in s:
        s.append(world)
    return sorted(set(s))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sizes", default="")
    ap.add_argument("--churn-size", type=int, default=2)
    ap.add_argument("--churn-count", type=int, default=16)
    ap.add_argument("--max-collective-mb", type=int, default=64)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    dev = torch.device(f"cuda:{local_rank}")
    dist.init_process_group(backend="nccl")

    # warm CUDA ctx + caching allocator + default NCCL comm (isolate GFC costs).
    w = torch.ones(1, device=dev)
    dist.all_reduce(w)
    _sync(dev)

    from gfc import SymmetricCollectiveConfig, SymmetricCollectiveRuntime

    sizes = ([int(x) for x in args.sizes.split(",") if x.strip()]
             if args.sizes else default_sizes(world))
    churn_size = min(args.churn_size, world)

    config = SymmetricCollectiveConfig(
        max_group_size=world,
        max_collective_bytes=args.max_collective_mb * 1024 * 1024,
        enable_debug_checks=False,
    )

    # ---- one-time runtime init (symm-mem rendezvous + region alloc) ----
    dist.barrier()
    free_before_rt = _free_mb(dev)
    t0 = time.perf_counter()
    runtime = SymmetricCollectiveRuntime(config, device=dev)
    _sync(dev)
    init_ms = (time.perf_counter() - t0) * 1e3
    dist.barrier()
    free_after_rt = _free_mb(dev)
    rt_mem_mb = free_before_rt - free_after_rt

    # warm up the all_gather kernel (Triton compile etc.) on the full group.
    warm_desc = runtime.register_group(tuple(range(world)), group_id=999_999)
    _all_gather(runtime, warm_desc, dev)
    _sync(dev)

    results = {
        "world": world, "backend": "gfc",
        "device_name": torch.cuda.get_device_name(dev),
        "max_collective_mb": args.max_collective_mb,
        "runtime_init_ms": init_ms, "runtime_init_mem_mb": rt_mem_mb,
        "sizes": sizes, "churn_size": churn_size, "churn_count": args.churn_count,
        "size_sweep": [], "churn": [],
    }
    if rank == 0:
        print(f"[setup] world={world} dev={results['device_name']} "
              f"max_coll={args.max_collective_mb}MB", flush=True)
        print(f"[init ] runtime ctor (ONE-TIME, symm rendezvous): "
              f"{init_ms:.3f}ms  symm_region_mem={rt_mem_mb:.1f}MB", flush=True)

    # ---- size sweep: register_group(range(s)) ----
    gid = 1000
    for s in sizes:
        ranks = tuple(range(s))
        is_member = rank in ranks
        dist.barrier()
        free_b = _free_mb(dev)
        t0 = time.perf_counter()
        desc = runtime.register_group(ranks, group_id=gid)   # the "add a map key"
        reg_ms = (time.perf_counter() - t0) * 1e3
        gid += 1
        first_ms = warm_ms = float("nan")
        if is_member:
            first_ms = _all_gather(runtime, desc, dev)
            warm_ms = _all_gather(runtime, desc, dev)
        dist.barrier()
        free_a = _free_mb(dev)
        m = {"size": s, "is_member": is_member, "register_ms": reg_ms,
             "first_coll_ms": first_ms, "warm_coll_ms": warm_ms,
             "mem_free_drop_mb": free_b - free_a}
        allm = _gather(m)
        if rank == 0:
            reg = [x["register_ms"] for x in allm]
            mem = [x["mem_free_drop_mb"] for x in allm]
            fc = [x["first_coll_ms"] for x in allm if x["is_member"]]
            wc = [x["warm_coll_ms"] for x in allm if x["is_member"]]
            agg = {"size": s, "register_ms_max": _mx(reg), "register_ms_mean": _mean(reg),
                   "first_coll_ms_max": _mx(fc), "warm_coll_ms_max": _mx(wc),
                   "mem_free_drop_mb_max": _mx(mem)}
            results["size_sweep"].append(agg)
            print(f"[size ] s={s:<2d} register={agg['register_ms_max']*1000:8.2f}us  "
                  f"first_coll={agg['first_coll_ms_max']:6.3f}ms  "
                  f"warm={agg['warm_coll_ms_max']:6.3f}ms  "
                  f"mem_drop={agg['mem_free_drop_mb_max']:6.1f}MB", flush=True)

    # ---- churn: K register_group() back-to-back ----
    dist.barrier()
    base_free = _free_mb(dev)
    for i in range(args.churn_count):
        t0 = time.perf_counter()
        runtime.register_group(tuple(range(churn_size)), group_id=10_000 + i)
        reg_ms = (time.perf_counter() - t0) * 1e3
        cum = base_free - _free_mb(dev)
        m = {"register_ms": reg_ms, "cumulative_drop_mb": cum}
        allm = _gather(m)
        if rank == 0:
            agg = {"index": i,
                   "register_ms_max": _mx([x["register_ms"] for x in allm]),
                   "cumulative_drop_mb": _mx([x["cumulative_drop_mb"] for x in allm])}
            results["churn"].append(agg)
            if i < 3 or i == args.churn_count - 1:
                print(f"[churn] #{i:<2d} register={agg['register_ms_max']*1000:8.2f}us  "
                      f"cumulative_mem={agg['cumulative_drop_mb']:6.1f}MB", flush=True)

    if rank == 0:
        print("\n== GFC group creation (vs NCCL: 0.5ms + 0.2-1.0s lazy init + 520MB/comm) ==")
        print(f"{'size':>4} {'register_us':>12} {'first_coll_ms':>14} {'warm_ms':>8} {'mem_MB':>7}")
        for a in results["size_sweep"]:
            print(f"{a['size']:>4} {a['register_ms_max']*1000:>12.2f} "
                  f"{a['first_coll_ms_max']:>14.3f} {a['warm_coll_ms_max']:>8.3f} "
                  f"{a['mem_free_drop_mb_max']:>7.1f}")
        if results["churn"]:
            regs = [a["register_ms_max"] for a in results["churn"]]
            last = results["churn"][-1]
            print(f"\nchurn {churn_size}-rank x {len(results['churn'])}: "
                  f"register mean={_mean(regs)*1000:.2f}us max={_mx(regs)*1000:.2f}us | "
                  f"cumulative mem over all groups = {last['cumulative_drop_mb']:.1f}MB "
                  f"(~{last['cumulative_drop_mb']/max(1,len(results['churn'])):.2f}MB/group)")
        print(f"\none-time runtime init: {init_ms:.3f}ms, symm region {rt_mem_mb:.1f}MB "
              f"(shared by ALL groups; max_collective_bytes={args.max_collective_mb}MB)")
        if args.out:
            with open(args.out, "w") as f:
                json.dump(results, f, indent=2)
            print(f"[out] wrote {args.out}", flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
