#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Numerical-equivalence check for distributed (patch/tile-parallel) VAE decode.

Decodes the SAME seeded latent two ways on the same ranks and compares rank 0's
output:
  * distributed  : set_parallel_size(world)  -> H/W tiles scattered across ranks
  * single-tiled : set_parallel_size(1)       -> diffusers tiled decode on 1 rank

The distributed path uses overlapping tiles + post-decode blend (no inter-rank
halo exchange), so it should match single-rank *tiled* decode up to fp noise.
As an extra reference we also (best-effort) compare against the non-tiled full
decode, which is the tiling approximation baseline (may OOM on large shapes).

  torchrun --standalone --nproc_per_node=4 \
      benchmarks/diffusion/vae_patch_parallel_equiv_check.py \
      --height 480 --width 832 --num-frames 81
"""

from __future__ import annotations

import argparse
import os

import torch
import torch.distributed as dist


def _init_distributed(master_port: int) -> tuple[int, int, torch.device]:
    if not torch.cuda.is_available():
        raise RuntimeError("requires CUDA")
    if "RANK" not in os.environ:
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("LOCAL_RANK", "0")
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", str(master_port))
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    from vllm_omni.diffusion.distributed.parallel_state import (
        init_distributed_environment,
        initialize_model_parallel,
    )

    init_distributed_environment(world_size=world, rank=rank)
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


def _metrics(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    a = a.float()
    b = b.float()
    diff = (a - b).abs()
    mse = torch.mean((a - b) ** 2).item()
    peak = max(a.abs().max().item(), b.abs().max().item(), 1e-8)
    psnr = float("inf") if mse == 0 else 10.0 * torch.log10(torch.tensor(peak * peak / mse)).item()
    return {
        "max_abs": diff.max().item(),
        "mean_abs": diff.mean().item(),
        "psnr_db": psnr,
        "shape_match": tuple(a.shape) == tuple(b.shape),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Wan-AI/Wan2.2-TI2V-5B-Diffusers")
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width", type=int, default=832)
    ap.add_argument("--num-frames", type=int, default=81)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--master-port", type=int, default=29556)
    ap.add_argument("--check-full", action="store_true", help="also compare vs non-tiled full decode (may OOM)")
    args = ap.parse_args()

    rank, world, device = _init_distributed(args.master_port)
    if world < 2:
        if rank == 0:
            print("WARN: world<2 — distributed path will fall back to single-rank; launch with torchrun --nproc_per_node>=2")

    from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_wan import (
        DistributedAutoencoderKLWan,
    )

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    vae = DistributedAutoencoderKLWan.from_pretrained(args.model, subfolder="vae", torch_dtype=dtype)
    vae = vae.to(device).eval()
    vae.use_tiling = True

    cfg = vae.config
    z_dim = int(cfg.z_dim)
    spatial = int(cfg.scale_factor_spatial)
    temporal = int(cfg.scale_factor_temporal)
    nlat_f = (args.num_frames - 1) // temporal + 1
    lat_h = args.height // spatial
    lat_w = args.width // spatial

    # Identical latent on every rank (same seed + shape + device kind).
    torch.manual_seed(args.seed)
    z = torch.randn((1, z_dim, nlat_f, lat_h, lat_w), device=device, dtype=dtype)
    if rank == 0:
        print(f"latent {tuple(z.shape)}  (H={args.height} W={args.width} F={args.num_frames})", flush=True)

    with torch.inference_mode():
        # distributed (PP=world)
        vae.set_parallel_size(world)
        out_dist = vae.tiled_decode(z, return_dict=False)[0]
        if dist.is_initialized():
            dist.barrier()
        # single-rank tiled
        vae.set_parallel_size(1)
        out_single = vae.tiled_decode(z, return_dict=False)[0]

        out_full = None
        if args.check_full:
            try:
                vae.use_tiling = False
                out_full = vae.decode(z, return_dict=False)[0]
            except Exception as exc:  # noqa: BLE001
                if rank == 0:
                    print(f"full decode skipped: {type(exc).__name__}: {exc}", flush=True)
            finally:
                vae.use_tiling = True

    if rank == 0:
        print(f"out_dist shape={tuple(out_dist.shape)} out_single shape={tuple(out_single.shape)}", flush=True)
        m = _metrics(out_dist, out_single)
        print(
            f"[distributed vs single-tiled]  shape_match={m['shape_match']}  "
            f"max|Δ|={m['max_abs']:.3e}  mean|Δ|={m['mean_abs']:.3e}  PSNR={m['psnr_db']:.1f} dB",
            flush=True,
        )
        if out_full is not None:
            mf = _metrics(out_dist, out_full)
            ms = _metrics(out_single, out_full)
            print(f"[distributed vs full ]  max|Δ|={mf['max_abs']:.3e}  PSNR={mf['psnr_db']:.1f} dB", flush=True)
            print(f"[single-tiled vs full]  max|Δ|={ms['max_abs']:.3e}  PSNR={ms['psnr_db']:.1f} dB", flush=True)
            print("(tiling-vs-full diff is the tiling approximation; should be ~equal for dist and single)", flush=True)

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
