"""Shared helpers for torchrun-based multi-process tests."""

from __future__ import annotations

import os
import sys
from typing import Optional

import torch
import torch.distributed as dist


def in_torchrun() -> bool:
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


def setup_nccl(timeout_seconds: int = 120) -> torch.device:
    """Initialize NCCL from torchrun env vars; return per-rank CUDA device."""
    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            timeout=__import__("datetime").timedelta(seconds=timeout_seconds),
        )
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    return device


def teardown() -> None:
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def rank0_print(*args, **kwargs) -> None:
    if int(os.environ.get("RANK", 0)) == 0:
        print(*args, **kwargs, flush=True)


def rank_print(*args, **kwargs) -> None:
    print(f"[rank {os.environ.get('RANK', '?')}]", *args, **kwargs, flush=True)


def order_after_caller(runtime) -> None:
    """Make ``runtime.stream`` wait for the work already queued on the caller's
    current stream, so a collective sees fully written inputs.

    This GFC version leaves that ordering to the caller (``wait_for_external``);
    Plectra does it in vllm_omni/diffusion/distributed/collective_runtime.py.
    Without it, a test that fills its input on the default stream races the
    collective on ``runtime.stream`` and can read stale bytes.
    """
    ev = torch.cuda.Event()
    ev.record()
    runtime.wait_for_external(ev)
