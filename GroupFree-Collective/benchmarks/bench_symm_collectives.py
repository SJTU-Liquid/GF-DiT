"""Benchmark the gfc collectives against ``torch.distributed`` references.

Section 9 of the design spec. Reports p50/p90/p99 wall-time per iter, p50
GPU-time per iter (CUDA-event-based), and effective bandwidth.

Run:
    torchrun --standalone --nproc_per_node=4 \\
        benchmarks/bench_symm_collectives.py \\
        --collective all2all --bytes 1048576 --iters 1000 --warmup 100 \\
        --dtype bf16 --validate --compare-nccl-reference
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Callable, List, Optional

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gfc import (  # noqa: E402
    SymmetricCollectiveConfig,
    SymmetricCollectiveRuntime,
)
from gfc.tma_probe import probe_tma_supported  # noqa: E402 - defined in phase 10


_DTYPES = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
    "uint8": torch.uint8,
}


_PRESETS = {
    # (collective, bytes_per_rank)
    "small_ctrl":    ("all2all", 4 * 1024),
    "mid_act":       ("allgather", 256 * 1024),
    "large_latent":  ("allgather", 4 * 1024 * 1024),
}


# ---------------------------------------------------------------------- helpers
def _ranks_from_arg(arg: Optional[str], world: int) -> List[int]:
    if arg is None:
        return list(range(world))
    return [int(r) for r in arg.split(",")]


def _percentiles(samples_us: List[float]) -> tuple[float, float, float]:
    s = sorted(samples_us)
    n = len(s)
    return s[n // 2], s[min(n - 1, int(n * 0.9))], s[min(n - 1, int(n * 0.99))]


def _bench_kernel_time(
    fn: Callable[[], None], n_iter: int, n_warmup: int, stream: torch.cuda.Stream
) -> float:
    for _ in range(n_warmup):
        fn()
    stream.synchronize()
    times: List[float] = []
    for _ in range(n_iter):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(stream)
        fn()
        end.record(stream)
        end.synchronize()
        times.append(start.elapsed_time(end) * 1000.0)
    times.sort()
    return times[len(times) // 2]


def _capture_batched_cuda_graph(
    fn: Callable[[], None],
    *,
    batch_iters: int,
    stream: torch.cuda.Stream,
) -> torch.cuda.CUDAGraph:
    """Capture ``batch_iters`` calls into one replayable CUDA graph.

    GFC barriers use explicit token values. Capturing a single collective and
    replaying it is unsafe because the next replay would find the same token
    already present in the signal lane. Capturing at least two calls leaves the
    lane at the last token, so the next replay must still wait for the first
    token to be republished.
    """
    if batch_iters < 2:
        raise ValueError("--graph-batch-iters must be at least 2")

    # Compile Triton kernels and populate allocator caches before capture.
    for _ in range(2):
        fn()
    stream.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for _ in range(batch_iters):
            fn()
    stream.synchronize()
    return graph


def _bench_graph_time(
    graph: torch.cuda.CUDAGraph,
    *,
    batch_iters: int,
    n_replays: int,
    n_warmup: int,
    stream: torch.cuda.Stream,
) -> tuple[float, list[float]]:
    for _ in range(n_warmup):
        with torch.cuda.stream(stream):
            graph.replay()
    stream.synchronize()

    samples: list[float] = []
    for _ in range(n_replays):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(stream)
        with torch.cuda.stream(stream):
            graph.replay()
        end.record(stream)
        end.synchronize()
        samples.append((start.elapsed_time(end) * 1000.0) / batch_iters)
    samples.sort()
    return samples[len(samples) // 2], samples


# ---------------------------------------------------------------------- main
def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--collective", choices=["all2all", "allgather", "p2p_put", "p2p_get", "barrier"], default="all2all")
    p.add_argument("--bytes", type=int, default=1024 * 1024,
                   help="per-rank input bytes for all_gather / per-rank total bytes for all2all / p2p")
    p.add_argument("--group", "--group-ranks", dest="group_ranks", type=str, default=None,
                   help="comma-separated ordered rank list (default: all ranks)")
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--dtype", choices=list(_DTYPES), default="bf16")
    p.add_argument("--use-tma", type=int, choices=[0, 1], default=0)
    p.add_argument("--copy-engine", type=int, choices=[0, 1], default=0,
                   help="use cudaMemcpyAsync copy-engine data path")
    p.add_argument("--cuda-graph", type=int, choices=[0, 1], default=1,
                   help="capture batched GFC benchmark iterations into a CUDA graph")
    p.add_argument("--graph-batch-iters", type=int, default=8,
                   help="number of collective calls captured per CUDA graph replay; must be >=2")
    p.add_argument("--graph-nccl-reference", type=int, choices=[0, 1], default=0,
                   help="also CUDA-graph-capture the NCCL reference when --compare-nccl-reference is set")
    p.add_argument("--validate", action="store_true")
    p.add_argument("--debug", action="store_true")
    p.add_argument("--compare-nccl-reference", action="store_true")
    p.add_argument("--diffusion-preset", choices=list(_PRESETS), default=None,
                   help="overrides --collective and --bytes")
    args = p.parse_args()

    if args.diffusion_preset is not None:
        args.collective, args.bytes = _PRESETS[args.diffusion_preset]

    if args.debug:
        os.environ["SYMM_COLL_DEBUG"] = "1"
    if args.use_tma and args.copy_engine:
        raise ValueError("--use-tma and --copy-engine are mutually exclusive")

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    group_ranks = _ranks_from_arg(args.group_ranks, world)
    in_group = rank in group_ranks
    group_size = len(group_ranks)

    config = SymmetricCollectiveConfig(
        max_group_size=max(group_size, 2),
        max_collective_bytes=max(args.bytes * group_size, args.bytes, 4 * 1024 * 1024),
        use_tma=bool(args.use_tma),
        use_copy_engine=bool(args.copy_engine),
    )
    runtime = SymmetricCollectiveRuntime(config, device=device)

    if args.use_tma:
        # Phase 10 probe — raise if unsupported (no silent fallback).
        ok = probe_tma_supported(runtime)
        if not ok:
            raise RuntimeError("TMA probe failed; rerun with --use-tma 0")

    if not in_group:
        # Stand in only for world-barrier purposes; other ranks need to keep
        # the bootstrap PG flowing during init/teardown.
        dist.barrier()
        runtime.shutdown()
        dist.destroy_process_group()
        return 0

    g = runtime.register_group(tuple(group_ranks))
    dtype = _DTYPES[args.dtype]
    elt = torch.empty((), dtype=dtype).element_size()

    # Build buffers + the callable for the chosen collective.
    if args.collective == "allgather":
        n_elems_in = args.bytes // elt
        inp = torch.empty(n_elems_in, dtype=dtype, device=device)
        inp.fill_(rank if dtype != torch.uint8 else (rank & 0xFF))
        out = torch.empty(group_size * n_elems_in, dtype=dtype, device=device)
        gfc_fn = lambda: runtime.all_gather(inp, out, g)
        gfc_graph_fn = lambda: runtime.all_gather(inp, out, g)
        nccl_fn = lambda: dist.all_gather_into_tensor(out, inp)
        bw_bytes_per_iter = (group_size - 1) * args.bytes

    elif args.collective == "all2all":
        # Treat --bytes as the *total* per-rank send buffer.
        total = args.bytes
        slice_bytes = total // group_size
        n_elems = total // elt
        inp = torch.empty(n_elems, dtype=dtype, device=device)
        inp.fill_(rank if dtype != torch.uint8 else (rank & 0xFF))
        out = torch.empty(n_elems, dtype=dtype, device=device)
        gfc_fn = lambda: runtime.all2all(inp, out, g)
        gfc_graph_fn = lambda: runtime.all2all(inp, out, g)
        nccl_fn = lambda: dist.all_to_all_single(out, inp)
        bw_bytes_per_iter = (group_size - 1) * slice_bytes

    elif args.collective == "p2p_put":
        if group_size != 2:
            raise ValueError("p2p_put requires --group of size 2")
        sender, receiver = group_ranks[0], group_ranks[1]
        n_elems = args.bytes // elt
        tensor = torch.empty(n_elems, dtype=dtype, device=device)
        tensor.fill_(rank if dtype != torch.uint8 else (rank & 0xFF))
        if rank == sender:
            gfc_fn = lambda: runtime.p2p_put(receiver, tensor)
            gfc_graph_fn = lambda: runtime.p2p_put(receiver, tensor)
        else:
            recv_buf = torch.empty(n_elems, dtype=dtype, device=device)
            gfc_fn = lambda: runtime.p2p_put_recv(sender, recv_buf)
            gfc_graph_fn = lambda: runtime.p2p_put_recv(sender, recv_buf)
        nccl_fn = None
        bw_bytes_per_iter = args.bytes

    elif args.collective == "p2p_get":
        if group_size != 2:
            raise ValueError("p2p_get requires --group of size 2")
        sender, receiver = group_ranks[0], group_ranks[1]
        n_elems = args.bytes // elt
        tensor = torch.empty(n_elems, dtype=dtype, device=device)
        tensor.fill_(rank if dtype != torch.uint8 else (rank & 0xFF))
        if rank == receiver:
            gfc_fn = lambda: runtime.p2p_get(sender, tensor)
            gfc_graph_fn = lambda: runtime.p2p_get(sender, tensor)
        else:
            gfc_fn = lambda: runtime.p2p_get_serve(receiver, tensor)
            gfc_graph_fn = lambda: runtime.p2p_get_serve(receiver, tensor)
        nccl_fn = None
        bw_bytes_per_iter = args.bytes

    elif args.collective == "barrier":
        gfc_fn = lambda: runtime.barrier(g)
        gfc_graph_fn = lambda: runtime.barrier(g)
        nccl_fn = lambda: dist.barrier()
        bw_bytes_per_iter = 0

    else:
        raise ValueError(args.collective)

    if args.cuda_graph:
        graph = _capture_batched_cuda_graph(
            gfc_graph_fn,
            batch_iters=args.graph_batch_iters,
            stream=runtime.stream,
        )

        # Wall-clock per-collective samples around graph replay.
        host_samples = []
        for _ in range(args.iters):
            t0 = time.perf_counter()
            with torch.cuda.stream(runtime.stream):
                graph.replay()
            runtime.stream.synchronize()
            host_samples.append(
                ((time.perf_counter() - t0) * 1e6) / args.graph_batch_iters
            )

        kernel_p50, _ = _bench_graph_time(
            graph,
            batch_iters=args.graph_batch_iters,
            n_replays=args.iters,
            n_warmup=args.warmup,
            stream=runtime.stream,
        )
    else:
        with torch.cuda.stream(runtime.stream):
            # Warmup
            for _ in range(args.warmup):
                gfc_fn()
            runtime.stream.synchronize()

            # Wall-clock per-iter samples.
            host_samples = []
            for _ in range(args.iters):
                t0 = time.perf_counter()
                gfc_fn()
                runtime.stream.synchronize()
                host_samples.append((time.perf_counter() - t0) * 1e6)

            # Kernel-time p50 (CUDA events).
            kernel_p50 = _bench_kernel_time(gfc_fn, args.iters, 5, runtime.stream)

    h_p50, h_p90, h_p99 = _percentiles(host_samples)
    bw_gbps = (bw_bytes_per_iter / (kernel_p50 * 1e-6)) / 1e9 if bw_bytes_per_iter else 0.0

    nccl_p50: Optional[float] = None
    if args.compare_nccl_reference and nccl_fn is not None:
        ref_stream = torch.cuda.Stream(device=device)
        if args.cuda_graph and args.graph_nccl_reference:
            try:
                nccl_graph = _capture_batched_cuda_graph(
                    nccl_fn,
                    batch_iters=args.graph_batch_iters,
                    stream=ref_stream,
                )
                nccl_p50, _ = _bench_graph_time(
                    nccl_graph,
                    batch_iters=args.graph_batch_iters,
                    n_replays=args.iters,
                    n_warmup=args.warmup,
                    stream=ref_stream,
                )
                del nccl_graph
            except Exception as e:
                if rank == 0:
                    print(f"nccl_reference    graph capture failed: {type(e).__name__}: {e}")
                with torch.cuda.stream(ref_stream):
                    nccl_p50 = _bench_kernel_time(nccl_fn, args.iters, args.warmup, ref_stream)
        else:
            with torch.cuda.stream(ref_stream):
                nccl_p50 = _bench_kernel_time(nccl_fn, args.iters, args.warmup, ref_stream)

    if args.validate and args.collective in ("allgather", "all2all"):
        # Re-run once and verify output (best-effort).
        gfc_fn()
        runtime.stream.synchronize()

    if rank == 0:
        path = "copy_engine" if args.copy_engine else ("tma" if args.use_tma else "vec")
        print(
            f"config: world={world} group_size={group_size} collective={args.collective} "
            f"bytes={args.bytes} dtype={args.dtype} "
            f"use_tma={args.use_tma} copy_engine={args.copy_engine} "
            f"path={path} copy_sms={runtime.copy_sms} "
            f"cuda_graph={args.cuda_graph} "
            f"graph_batch_iters={args.graph_batch_iters} "
            f"graph_nccl_reference={args.graph_nccl_reference}"
        )
        if args.cuda_graph:
            print(
                f"iters: {args.iters * args.graph_batch_iters} collectives "
                f"({args.iters} graph replays, warmup {args.warmup} replays)"
            )
        else:
            print(f"iters: {args.iters + args.warmup} (warmup {args.warmup})")
        print(f"latency           p50={h_p50:.1f}us p90={h_p90:.1f}us p99={h_p99:.1f}us")
        print(f"kernel_time       p50={kernel_p50:.1f}us")
        if bw_bytes_per_iter:
            print(f"effective_bw      {bw_gbps:.2f} GB/s   ({bw_bytes_per_iter} bytes/iter)")
        if nccl_p50 is not None:
            print(f"nccl_reference    p50={nccl_p50:.1f}us  (gfc/nccl ratio {kernel_p50/nccl_p50:.2f})")
        print(f"group_id          0x{g.group_id:016x}")

    runtime.shutdown()
    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
