#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Standalone VAE decode latency profiler under patch/tile parallelism.

Measures Wan2.2 VAE ``decode`` latency at a given VAE patch-parallel degree
(= the launched world size), so the motivation figure can show how the VAE
stage scales with the execution group size.

Why a separate script: the served Wan2.2-TI2V-5B pipeline wires in the plain
diffusers ``AutoencoderKLWan`` (single-rank), so the main stage profiler treats
VAE decode as parallelism-invariant. Here we instead load
``DistributedAutoencoderKLWan``, which spatially tiles the 5D video latent
(B, C, T, H, W) along H/W and load-balances the tiles across the DiT group.

For an apples-to-apples scaling curve we always call ``tiled_decode``: at PP=1
it falls back to single-rank tiled decode, at PP>1 it runs the distributed
tile executor over the DiT group.

Launch one process group per parallel degree (PP = world size):

  # PP=1 (single-GPU baseline; no torchrun needed)
  .venv/bin/python benchmarks/diffusion/vae_patch_parallel_profiler.py \
      --out out/vae/pp1.json

  # PP=4
  .venv/bin/torchrun --standalone --nproc_per_node=4 \
      benchmarks/diffusion/vae_patch_parallel_profiler.py \
      --out out/vae/pp4.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist


@dataclass
class GridPoint:
    height: int
    width: int
    num_frames: int


def _default_grid() -> list[GridPoint]:
    # Shapes large enough to produce several VAE tiles (so PP>1 has work to
    # split). Frames are kept == 1 (mod temporal) so the latent-frame count is
    # integral. latent dims = pixel // spatial_scale.
    return [
        GridPoint(480, 832, 81),
        GridPoint(720, 1280, 49),
        GridPoint(720, 1280, 81),
    ]


def _parse_grid(spec_json: str | None) -> list[GridPoint]:
    if not spec_json:
        return _default_grid()
    spec = json.loads(spec_json)
    return [GridPoint(int(p["height"]), int(p["width"]), int(p["num_frames"])) for p in spec]


def _init_distributed(args: argparse.Namespace) -> tuple[int, int, torch.device]:
    if not torch.cuda.is_available():
        raise RuntimeError("VAE profiler requires CUDA")

    # Allow running PP=1 without torchrun.
    if "RANK" not in os.environ:
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("LOCAL_RANK", "0")
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", str(args.master_port))

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))

    if torch.cuda.device_count() <= local_rank:
        raise RuntimeError(
            f"local_rank={local_rank} exceeds cuda_device_count={torch.cuda.device_count()}"
        )
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    from vllm_omni.diffusion.distributed.parallel_state import (
        init_distributed_environment,
        initialize_model_parallel,
    )

    init_distributed_environment(world_size=world, rank=rank)
    # tp=cfg=ring=1, sp=world -> the DiT group spans all ranks, which is the
    # group the distributed VAE tiles across (DistributedVaeExecutor uses
    # get_dit_group()).
    initialize_model_parallel(
        tensor_parallel_size=1,
        sequence_parallel_size=world,
        ulysses_degree=world,
        ring_degree=1,
        cfg_parallel_size=1,
        data_parallel_size=1,
        pipeline_parallel_size=1,
    )
    return rank, world, device


def _load_vae(args: argparse.Namespace, world: int, device: torch.device) -> Any:
    from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_wan import (
        DistributedAutoencoderKLWan,
    )

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    # from_pretrained calls init_distributed() internally, which builds the
    # DistributedVaeExecutor over the (already-initialized) DiT group.
    vae = DistributedAutoencoderKLWan.from_pretrained(
        args.model, subfolder="vae", torch_dtype=dtype
    )
    vae = vae.to(device).eval()
    vae.use_tiling = not args.no_tiling
    vae.set_parallel_size(world)
    return vae


def _latent_for(vae: Any, point: GridPoint, device: torch.device) -> torch.Tensor:
    cfg = vae.config
    z_dim = int(cfg.z_dim)
    spatial = int(cfg.scale_factor_spatial)
    temporal = int(cfg.scale_factor_temporal)
    num_latent_frames = (point.num_frames - 1) // temporal + 1
    lat_h = point.height // spatial
    lat_w = point.width // spatial
    return torch.randn(
        (1, z_dim, num_latent_frames, lat_h, lat_w),
        device=device,
        dtype=vae.dtype,
    )


def _allreduce_max(value_ms: float, device: torch.device) -> float:
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return value_ms
    t = torch.tensor([float(value_ms)], dtype=torch.float64, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return float(t.item())


def _time_decode(vae: Any, z: torch.Tensor, args: argparse.Namespace, device: torch.device) -> list[float]:
    samples_ms: list[float] = []
    with torch.inference_mode():
        for i in range(args.warmup_iters + args.bench_iters):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize(device)
            if dist.is_initialized():
                dist.barrier()
            start.record()
            if args.no_tiling:
                # native (non-tiled) full decode — single-GPU baseline.
                vae.decode(z, return_dict=False)
            else:
                # tiled_decode routes to the distributed tile executor when PP>1,
                # else to single-rank diffusers tiled decode.
                vae.tiled_decode(z, return_dict=False)
            end.record()
            torch.cuda.synchronize(device)
            dt = _allreduce_max(start.elapsed_time(end), device)
            if i >= args.warmup_iters:
                samples_ms.append(dt)
    return samples_ms


def _summary(samples_ms: list[float]) -> dict[str, float]:
    s = sorted(samples_ms)
    return {
        "latency_ms_mean": statistics.fmean(s),
        "latency_ms_median": statistics.median(s),
        "latency_ms_min": s[0],
        "latency_ms_max": s[-1],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Wan-AI/Wan2.2-TI2V-5B-Diffusers")
    ap.add_argument("--grid-json", default=None, help="JSON list of {height,width,num_frames}")
    ap.add_argument("--warmup-iters", type=int, default=3)
    ap.add_argument("--bench-iters", type=int, default=10)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--out", required=True, help="output JSON path (written by rank 0)")
    ap.add_argument("--no-tiling", action="store_true",
                    help="disable tiling and time native full vae.decode (single-GPU baseline)")
    ap.add_argument("--master-port", type=int, default=29555)
    args = ap.parse_args()

    rank, world, device = _init_distributed(args)
    grid = _parse_grid(args.grid_json)
    vae = _load_vae(args, world, device)

    cfg = vae.config
    spatial = int(cfg.scale_factor_spatial)
    temporal = int(cfg.scale_factor_temporal)

    results: list[dict[str, Any]] = []
    for point in grid:
        z = _latent_for(vae, point, device)
        if rank == 0:
            print(
                f"[pp={world}] H={point.height} W={point.width} F={point.num_frames} "
                f"-> latent {tuple(z.shape)}",
                flush=True,
            )
        samples_ms = _time_decode(vae, z, args, device)
        if rank == 0:
            row = {
                "height": point.height,
                "width": point.width,
                "num_frames": point.num_frames,
                "voxels": point.height * point.width * point.num_frames,
                "latent_shape": list(z.shape),
                **_summary(samples_ms),
            }
            results.append(row)
            print(f"    median={row['latency_ms_median']:.2f}ms mean={row['latency_ms_mean']:.2f}ms", flush=True)

    if rank == 0:
        out = {
            "model": args.model,
            "vae_patch_parallel_size": world,
            "decode_mode": "native" if args.no_tiling else "tiled",
            "dtype": args.dtype,
            "warmup_iters": args.warmup_iters,
            "bench_iters": args.bench_iters,
            "vae_scale_factor_spatial": spatial,
            "vae_scale_factor_temporal": temporal,
            "samples": results,
        }
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=2))
        print(f"wrote {out_path}", flush=True)

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
