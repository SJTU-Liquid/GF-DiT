#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Benchmark the cost of creating an NCCL subgroup.

Launch on N GPUs (auto world size):

    torchrun --standalone --nproc_per_node=6 \
        benchmarks/distributed/nccl_subgroup_cost.py

Per subgroup creation it measures:

  * newgroup_cpu_ms  : host wall time of ``dist.new_group()`` itself. This is a
                       world-wide call -- *every* rank (members and non-members)
                       must invoke it -- so it is timed on all ranks.
  * first_coll_ms    : wall time of the FIRST all_reduce on the subgroup,
                       bracketed by cuda.synchronize(). NCCL inits the
                       communicator lazily (ncclCommInitRank: bootstrap +
                       channel setup + buffer alloc) on first use, so this is
                       where the real init cost lands.
  * warm_coll_ms     : a second all_reduce, for a warm reference baseline.
  * mem_free_drop_mb : driver-level free-memory delta from
                       ``torch.cuda.mem_get_info`` on the local device. This
                       catches the NCCL communicator buffers, which live outside
                       torch's caching allocator.

Two sweeps:

  size  : one group per size ``s`` in --sizes, over ranks [0..s-1].
  churn : --churn-count groups of --churn-size back-to-back, tracking the
          cumulative free-memory drop and per-group time stability (relevant to
          elastic SP serving that repeatedly churns subgroups).

Right after init a warmup all_reduce runs on the WORLD group so the default
communicator, CUDA context, and caching-allocator segment are all initialized
before any measurement -- otherwise the first subgroup would absorb that cost.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import torch
import torch.distributed as dist


def _sync(dev: torch.device) -> None:
    torch.cuda.synchronize(dev)


def _free_mb(dev: torch.device) -> float:
    free, _ = torch.cuda.mem_get_info(dev)
    return free / (1024 ** 2)


def _time_all_reduce(group, dev: torch.device) -> float:
    t = torch.ones(1, device=dev)
    _sync(dev)
    t0 = time.perf_counter()
    dist.all_reduce(t, group=group)
    _sync(dev)
    return (time.perf_counter() - t0) * 1e3


def create_one_group(ranks: list[int], rank: int, dev: torch.device) -> tuple[dict, object]:
    """Create a subgroup over ``ranks`` and measure its cost.

    All ranks in the world must call this in lockstep (new_group is collective).
    """
    is_member = rank in ranks

    dist.barrier()
    free_before = _free_mb(dev)

    t0 = time.perf_counter()
    group = dist.new_group(ranks=ranks)
    newgroup_cpu_ms = (time.perf_counter() - t0) * 1e3

    first_ms = warm_ms = float("nan")
    if is_member:
        first_ms = _time_all_reduce(group, dev)   # triggers lazy comm init
        warm_ms = _time_all_reduce(group, dev)

    dist.barrier()
    free_after = _free_mb(dev)

    metrics = {
        "rank": rank,
        "is_member": is_member,
        "newgroup_cpu_ms": newgroup_cpu_ms,
        "first_coll_ms": first_ms,
        "warm_coll_ms": warm_ms,
        "mem_free_drop_mb": free_before - free_after,
        "free_before_mb": free_before,
        "free_after_mb": free_after,
    }
    return metrics, group


def _gather(obj: dict) -> list[dict]:
    out: list = [None] * dist.get_world_size()
    dist.all_gather_object(out, obj)
    return out


def _mx(xs: list[float]) -> float:
    return max(xs) if xs else float("nan")


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def summarize(all_metrics: list[dict], size: int, index: int | None = None) -> dict:
    members = [m for m in all_metrics if m["is_member"]]
    ng = [m["newgroup_cpu_ms"] for m in all_metrics]
    fc = [m["first_coll_ms"] for m in members]
    wc = [m["warm_coll_ms"] for m in members]
    drop = [m["mem_free_drop_mb"] for m in members]
    cum = [m.get("cumulative_drop_mb") for m in members if m.get("cumulative_drop_mb") is not None]
    out = {
        "size": size,
        "index": index,
        "newgroup_cpu_ms_max": _mx(ng),
        "newgroup_cpu_ms_mean": _mean(ng),
        "first_coll_ms_max": _mx(fc),
        "first_coll_ms_mean": _mean(fc),
        "warm_coll_ms_max": _mx(wc),
        "mem_free_drop_mb_max": _mx(drop),
        "mem_free_drop_mb_mean": _mean(drop),
        # what the user cares about: full "until usable" cost = new_group + first collective
        "total_create_ms_max": _mx(ng) + _mx(fc),
    }
    if cum:
        out["cumulative_drop_mb"] = _mx(cum)
    return out


def default_sizes(world: int) -> list[int]:
    sizes: list[int] = []
    p = 2
    while p <= world:
        sizes.append(p)
        p *= 2
    if world >= 2 and world not in sizes:
        sizes.append(world)
    return sorted(set(sizes))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sizes", default="",
                    help="comma list of subgroup sizes; default powers-of-2 up to world + world")
    ap.add_argument("--churn-size", type=int, default=2, help="rank count for each churn group")
    ap.add_argument("--churn-count", type=int, default=16, help="number of churn groups to create")
    ap.add_argument("--out", default="", help="optional path to write JSON results (rank 0)")
    args = ap.parse_args()

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    dev = torch.device(f"cuda:{local_rank}")

    dist.init_process_group(backend="nccl")

    # Warm up the world communicator + CUDA context + allocator segment.
    warm = torch.ones(1, device=dev)
    dist.all_reduce(warm)
    _sync(dev)

    sizes = ([int(x) for x in args.sizes.split(",") if x.strip()]
             if args.sizes else default_sizes(world))
    churn_size = min(args.churn_size, world)

    results: dict = {
        "world": world,
        "backend": "nccl",
        "torch": torch.__version__,
        "nccl": ".".join(str(v) for v in torch.cuda.nccl.version()),
        "device_name": torch.cuda.get_device_name(dev),
        "sizes": sizes,
        "churn_size": churn_size,
        "churn_count": args.churn_count,
        "size_sweep": [],
        "churn": [],
    }

    if rank == 0:
        print(f"[setup] world={world} sizes={sizes} churn={churn_size}x{args.churn_count} "
              f"nccl={results['nccl']} dev={results['device_name']}", flush=True)

    # ---- size sweep ----
    for s in sizes:
        metrics, _group = create_one_group(list(range(s)), rank, dev)
        agg = summarize(_gather(metrics), s)
        if rank == 0:
            results["size_sweep"].append(agg)
            print(f"[size ] s={s:<2d} new_group={agg['newgroup_cpu_ms_max']:7.3f}ms  "
                  f"first_coll={agg['first_coll_ms_max']:7.3f}ms  "
                  f"warm={agg['warm_coll_ms_max']:6.3f}ms  "
                  f"total={agg['total_create_ms_max']:7.3f}ms  "
                  f"mem_drop={agg['mem_free_drop_mb_max']:7.1f}MB", flush=True)

    # ---- churn ----
    churn_ranks = list(range(churn_size))
    dist.barrier()
    base_free = _free_mb(dev)
    for i in range(args.churn_count):
        metrics, _group = create_one_group(churn_ranks, rank, dev)
        metrics["cumulative_drop_mb"] = base_free - _free_mb(dev)
        agg = summarize(_gather(metrics), churn_size, index=i)
        if rank == 0:
            results["churn"].append(agg)
            print(f"[churn] #{i:<2d} new_group={agg['newgroup_cpu_ms_max']:7.3f}ms  "
                  f"first_coll={agg['first_coll_ms_max']:7.3f}ms  "
                  f"mem_drop={agg['mem_free_drop_mb_max']:6.1f}MB  "
                  f"cumulative={agg.get('cumulative_drop_mb', float('nan')):7.1f}MB", flush=True)

    if rank == 0:
        if results["size_sweep"]:
            print("\n== size sweep (cost to a usable subgroup) ==")
            print(f"{'size':>4} {'new_group_ms':>13} {'first_coll_ms':>14} "
                  f"{'total_ms':>9} {'mem_MB':>8}")
            for a in results["size_sweep"]:
                print(f"{a['size']:>4} {a['newgroup_cpu_ms_max']:>13.3f} "
                      f"{a['first_coll_ms_max']:>14.3f} {a['total_create_ms_max']:>9.3f} "
                      f"{a['mem_free_drop_mb_max']:>8.1f}")
        if results["churn"]:
            first = results["churn"][0]
            last = results["churn"][-1]
            ng_vals = [a["newgroup_cpu_ms_max"] for a in results["churn"]]
            fc_vals = [a["first_coll_ms_max"] for a in results["churn"]]
            print(f"\n== churn ({churn_size}-rank x {len(results['churn'])}) ==")
            print(f"  new_group_ms : first={first['newgroup_cpu_ms_max']:.3f} "
                  f"last={last['newgroup_cpu_ms_max']:.3f} mean={_mean(ng_vals):.3f}")
            print(f"  first_coll_ms: first={first['first_coll_ms_max']:.3f} "
                  f"last={last['first_coll_ms_max']:.3f} mean={_mean(fc_vals):.3f}")
            print(f"  cumulative free-mem drop over {len(results['churn'])} groups: "
                  f"{last.get('cumulative_drop_mb', float('nan')):.1f}MB "
                  f"(~{last.get('cumulative_drop_mb', 0) / max(1, len(results['churn'])):.1f}MB/group)")

        if args.out:
            with open(args.out, "w") as f:
                json.dump(results, f, indent=2)
            print(f"\n[out] wrote {args.out}", flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
