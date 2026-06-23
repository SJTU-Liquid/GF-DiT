#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Microbenchmark the real runtime_v2 RESHARD path.

Run on four GPUs:

    torchrun --standalone --nproc_per_node=4 \
        benchmarks/diffusion/runtime_v2_reshard_microbench.py

This intentionally does not reimplement NCCL send/recv.  Each iteration seeds a
real Qwen-Image or Wan runtime state artifact, then drives the same worker-side
runtime_v2 migration phases used in serving:

    describe_layout -> build_schedule -> accept_schedule -> execute_transfer

The benchmark uses the production Qwen/Wan state codecs for tensor layouts.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import threading
import time
import uuid
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import torch
import torch.distributed as dist


GROUP_RANKS: dict[str, tuple[int, ...]] = {
    "SP1": (0,),
    "SP2": (0, 1),
    "SP4": (0, 1, 2, 3),
}

DEFAULT_TRANSITIONS = ("SP4->SP2", "SP2->SP4", "SP4->SP1", "SP1->SP4")
DEFAULT_CASES = (
    "image_latent_small",
    "image_latent_large",
    "video_latent_short",
    "video_latent_long",
)


@dataclass(frozen=True)
class CaseSpec:
    name: str
    model: str
    metadata: dict[str, Any]


@dataclass
class IterationMetrics:
    transition: str
    latent_name: str
    latent_shape: str
    bytes: int
    reshard_bytes: int
    reshard_time_ms: float
    artifact_materialize_time_ms: float
    artifact_shard_time_ms: float
    scheduler_overhead_ms: float
    task_launch_overhead_ms: float
    interprocess_transfer_time_ms: float
    group_switch_gap_ms: float


class _ResultCaptureRuntime:
    """Small in-process host for MigrationDataPlaneMixin.

    The methods and fields mirror the subset that _WorkerProcessRuntime exposes
    to MigrationDataPlaneMixin.  Migration itself still runs through the
    production mixin methods.
    """

    def __init__(self, *, worker_rank: int, groups: Mapping[str, Any], codec_registry: Any) -> None:
        self.worker_rank = worker_rank
        self.local_artifacts: dict[tuple[str, str, str], Any] = {}
        self._migration_schedules: dict[str, Any] = {}
        self._artifacts_lock = threading.RLock()
        self._execution_group_specs_by_id = dict(groups)
        self._parallel_config_by_group: dict[str, Any] = {}
        self.codec_registry = codec_registry
        self._results: list[Any] = []

    @staticmethod
    def _artifact_key(request_id: str, group_id: str, artifact_id: str) -> tuple[str, str, str]:
        return (request_id, group_id, artifact_id)

    def _group_spec(self, group_id: str) -> Any:
        return self._execution_group_specs_by_id[str(group_id)]

    def _maybe_activate_group_session(self, group_id: str) -> None:
        # Worker process sessions only set global parallel state around model
        # execution.  This harness supplies the needed SP group explicitly when
        # measuring materialization.
        return None

    def _send_result(self, result: Any) -> None:
        self._results.append(result)

    def pop_result(self) -> Any:
        if len(self._results) != 1:
            raise RuntimeError(f"expected exactly one migration result, got {len(self._results)}")
        return self._results.pop()


def _require_cuda() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        cuda_available = torch.cuda.is_available()
    if not cuda_available:
        raise RuntimeError("runtime_v2 reshard microbench requires CUDA/NCCL; no CUDA device is available")


def _load_runtime_symbols() -> dict[str, Any]:
    from vllm_omni.diffusion.runtime_v2 import data_plane as dp
    from vllm_omni.diffusion.runtime_v2.adapters.qwen_image import (
        QWEN_IMAGE_STATE_CODEC_ID,
        QwenImageRuntimeState,
        QwenImageStateLayoutCodec,
    )
    from vllm_omni.diffusion.runtime_v2.adapters.wan import (
        WAN_STATE_CODEC_ID,
        WanRuntimeState,
        WanStateLayoutCodec,
    )
    from vllm_omni.diffusion.runtime_v2.codec import ArtifactLayoutCodecRegistry
    from vllm_omni.diffusion.runtime_v2.multiproc_worker import MigrateArtifactsCommand
    from vllm_omni.diffusion.runtime_v2.protocol import (
        ArtifactHandle,
        ArtifactKind,
        ArtifactLayout,
        ArtifactValue,
        ExecutionGroupSpec,
        MigrateArtifactSpec,
        MigrateArtifactsPlan,
        ParallelSpec,
        TaskKind,
    )

    return {
        "ArtifactHandle": ArtifactHandle,
        "ArtifactKind": ArtifactKind,
        "ArtifactLayout": ArtifactLayout,
        "ArtifactLayoutCodecRegistry": ArtifactLayoutCodecRegistry,
        "ArtifactValue": ArtifactValue,
        "ExecutionGroupSpec": ExecutionGroupSpec,
        "MigrateArtifactSpec": MigrateArtifactSpec,
        "MigrateArtifactsCommand": MigrateArtifactsCommand,
        "MigrateArtifactsPlan": MigrateArtifactsPlan,
        "MigrationDataPlaneMixin": dp.MigrationDataPlaneMixin,
        "ParallelSpec": ParallelSpec,
        "QWEN_IMAGE_STATE_CODEC_ID": QWEN_IMAGE_STATE_CODEC_ID,
        "QwenImageRuntimeState": QwenImageRuntimeState,
        "QwenImageStateLayoutCodec": QwenImageStateLayoutCodec,
        "TaskKind": TaskKind,
        "WAN_STATE_CODEC_ID": WAN_STATE_CODEC_ID,
        "WanRuntimeState": WanRuntimeState,
        "WanStateLayoutCodec": WanStateLayoutCodec,
        "_MIGRATE_PHASE_ACCEPT_SCHEDULE": dp._MIGRATE_PHASE_ACCEPT_SCHEDULE,
        "_MIGRATE_PHASE_BUILD_SCHEDULE": dp._MIGRATE_PHASE_BUILD_SCHEDULE,
        "_MIGRATE_PHASE_DESCRIBE_LAYOUT": dp._MIGRATE_PHASE_DESCRIBE_LAYOUT,
        "_MIGRATE_PHASE_EXECUTE_TRANSFER": dp._MIGRATE_PHASE_EXECUTE_TRANSFER,
        "parse_torch_dtype": dp.parse_torch_dtype,
        "tensor_numel": dp.tensor_numel,
    }


def _qwen_latent_seq_len(height: int, width: int) -> int:
    return max(1, (int(height) // 16) * (int(width) // 16))


def _case_specs() -> dict[str, CaseSpec]:
    return {
        "image_latent_small": CaseSpec(
            name="image_latent_small",
            model="qwen_image",
            metadata={
                "height": 1024,
                "width": 1024,
                "text_seq_len": 512,
                "latent_seq_len": _qwen_latent_seq_len(1024, 1024),
                "num_steps": 1,
                "num_images_per_prompt": 1,
                "do_true_cfg": False,
                "output_type": "latent",
            },
        ),
        "image_latent_large": CaseSpec(
            name="image_latent_large",
            model="qwen_image",
            metadata={
                "height": 2048,
                "width": 2048,
                "text_seq_len": 512,
                "latent_seq_len": _qwen_latent_seq_len(2048, 2048),
                "num_steps": 1,
                "num_images_per_prompt": 1,
                "do_true_cfg": False,
                "output_type": "latent",
            },
        ),
        "video_latent_short": CaseSpec(
            name="video_latent_short",
            model="wan",
            metadata={
                "height": 480,
                "width": 832,
                "num_frames": 17,
                "text_seq_len": 512,
                "num_steps": 1,
                "num_outputs_per_prompt": 1,
                "do_true_cfg": False,
                "has_image": False,
                "output_type": "latent",
            },
        ),
        "video_latent_long": CaseSpec(
            name="video_latent_long",
            model="wan",
            metadata={
                "height": 720,
                "width": 1280,
                "num_frames": 81,
                "text_seq_len": 512,
                "num_steps": 1,
                "num_outputs_per_prompt": 1,
                "do_true_cfg": False,
                "has_image": False,
                "output_type": "latent",
            },
        ),
    }


def _parse_transition(raw: str) -> tuple[str, str]:
    try:
        src, dst = raw.split("->", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("transition must look like SP4->SP2") from exc
    src = src.strip().upper()
    dst = dst.strip().upper()
    if src not in GROUP_RANKS or dst not in GROUP_RANKS:
        raise argparse.ArgumentTypeError(f"unknown transition groups: {raw!r}")
    if src == dst:
        raise argparse.ArgumentTypeError("transition source and destination must differ")
    return src, dst


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * pct
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _fmt_shape(shape: tuple[int, ...]) -> str:
    return "x".join(str(dim) for dim in shape)


def _ns_to_ms(ns: int) -> float:
    return float(ns) / 1_000_000.0


def _sync(device: torch.device) -> None:
    torch.cuda.synchronize(device)


def _max_float(value: float, device: torch.device) -> float:
    tensor = torch.tensor([float(value)], dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return float(tensor.item())


def _init_distributed() -> tuple[int, int, torch.device]:
    _require_cuda()
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    if world < 4:
        raise RuntimeError(f"this benchmark requires at least 4 ranks, got world_size={world}")
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    if local_rank >= torch.cuda.device_count():
        raise RuntimeError(
            "runtime_v2 reshard microbench requires one CUDA device per local rank; "
            f"local_rank={local_rank}, cuda_device_count={torch.cuda.device_count()}"
        )
    torch.cuda.set_device(local_rank)
    return rank, world, torch.device("cuda", local_rank)


def _build_group_specs(sym: Mapping[str, Any]) -> dict[str, Any]:
    specs = {}
    for group_id, ranks in GROUP_RANKS.items():
        sp = len(ranks)
        specs[group_id] = sym["ExecutionGroupSpec"](
            group_id=group_id,
            ranks=ranks,
            parallel_spec=sym["ParallelSpec"](tp=1, sp=sp, cfg=1),
            supported_task_kinds=tuple(sym["TaskKind"]),
            ulysses_degree=sp,
            ring_degree=1,
        )
    return specs


def _create_dist_groups() -> dict[str, Any]:
    return {name: dist.new_group(ranks=list(ranks)) for name, ranks in GROUP_RANKS.items()}


def _fake_qwen_pipeline(dtype: torch.dtype, args: argparse.Namespace) -> Any:
    text_encoder = SimpleNamespace(
        dtype=dtype,
        config=SimpleNamespace(hidden_size=int(args.qwen_text_hidden_size)),
    )
    transformer = SimpleNamespace(
        dtype=dtype,
        in_channels=int(args.qwen_transformer_in_channels),
        guidance_embeds=False,
        config=SimpleNamespace(hidden_size=int(args.qwen_text_hidden_size)),
    )
    return SimpleNamespace(text_encoder=text_encoder, transformer=transformer, vae=SimpleNamespace(dtype=dtype))


def _fake_wan_pipeline(dtype: torch.dtype, args: argparse.Namespace) -> Any:
    text_encoder = SimpleNamespace(
        dtype=dtype,
        config=SimpleNamespace(hidden_size=int(args.wan_text_hidden_size)),
    )
    return SimpleNamespace(
        text_encoder=text_encoder,
        transformer=SimpleNamespace(dtype=dtype),
        transformer_2=None,
        transformer_config={
            "patch_size": (1, 2, 2),
            "in_channels": int(args.wan_latent_channels),
            "out_channels": int(args.wan_latent_channels),
        },
        vae=SimpleNamespace(dtype=dtype),
        vae_scale_factor_spatial=int(args.wan_vae_scale_factor_spatial),
        vae_scale_factor_temporal=int(args.wan_vae_scale_factor_temporal),
        expand_timesteps=False,
    )


def _build_codec_registry(sym: Mapping[str, Any], dtype: torch.dtype, args: argparse.Namespace) -> Any:
    registry = sym["ArtifactLayoutCodecRegistry"]()
    registry.register(sym["QwenImageStateLayoutCodec"](_fake_qwen_pipeline(dtype, args)))
    registry.register(sym["WanStateLayoutCodec"](_fake_wan_pipeline(dtype, args)))
    return registry


def _codec_id_for_case(case: CaseSpec, sym: Mapping[str, Any]) -> str:
    if case.model == "qwen_image":
        return sym["QWEN_IMAGE_STATE_CODEC_ID"]
    if case.model == "wan":
        return sym["WAN_STATE_CODEC_ID"]
    raise ValueError(f"unknown case model: {case.model!r}")


def _make_state(case: CaseSpec, *, device: torch.device, dtype: torch.dtype, sym: Mapping[str, Any]) -> Any:
    metadata = case.metadata
    if case.model == "qwen_image":
        return sym["QwenImageRuntimeState"](
            request=None,
            prompt="microbench",
            negative_prompt=None,
            height=int(metadata["height"]),
            width=int(metadata["width"]),
            num_steps=int(metadata["num_steps"]),
            output_type="latent",
            max_sequence_length=int(metadata["text_seq_len"]),
            device=device,
            dtype=dtype,
            guidance_scale=1.0,
            true_cfg_scale=1.0,
            do_true_cfg=bool(metadata.get("do_true_cfg", False)),
        )
    if case.model == "wan":
        return sym["WanRuntimeState"](
            request=None,
            prompt="microbench",
            negative_prompt=None,
            height=int(metadata["height"]),
            width=int(metadata["width"]),
            num_frames=int(metadata["num_frames"]),
            num_steps=int(metadata["num_steps"]),
            output_type="latent",
            device=device,
            dtype=dtype,
            guidance_low=1.0,
            guidance_high=1.0,
            boundary_timestep=0.875,
        )
    raise ValueError(f"unknown case model: {case.model!r}")


def _write_field(value: Any, field_path: tuple[str, ...], new_value: Any) -> None:
    cursor = value
    for part in field_path[:-1]:
        cursor = cursor[part] if isinstance(cursor, dict) else getattr(cursor, part)
    leaf = field_path[-1]
    if isinstance(cursor, dict):
        cursor[leaf] = new_value
    else:
        setattr(cursor, leaf, new_value)


def _fill_declared_tensors(
    *,
    value: Any,
    layout: Any,
    sym: Mapping[str, Any],
    device: torch.device,
) -> None:
    for field in layout.tensors:
        dtype = sym["parse_torch_dtype"](field.dtype)
        shape = tuple(int(dim) for dim in field.global_shape)
        tensor = torch.empty(shape, dtype=dtype, device=device)
        _write_field(value, tuple(field.field_path), tensor)


def _seed_source_artifact(
    *,
    runtime: Any,
    rank: int,
    case: CaseSpec,
    src_group: Any,
    handle: Any,
    device: torch.device,
    dtype: torch.dtype,
    sym: Mapping[str, Any],
) -> float:
    runtime.local_artifacts.clear()
    if rank not in src_group.ranks:
        return 0.0
    begin = time.perf_counter_ns()
    group_rank = src_group.ranks.index(rank)
    state = _make_state(case, device=device, dtype=dtype, sym=sym)
    layout = runtime.codec_registry.describe_output(
        group=src_group,
        group_rank=group_rank,
        request_metadata=case.metadata,
        artifact=handle,
    )
    _fill_declared_tensors(value=state, layout=layout, sym=sym, device=device)
    value = sym["ArtifactValue"](handle=handle, value=state)
    value = runtime._shard_output_artifact(
        artifact_value=value,
        group_id=src_group.group_id,
        request_metadata=case.metadata,
    )
    runtime.local_artifacts[runtime._artifact_key(handle.request_id, src_group.group_id, handle.artifact_id)] = value
    _sync(device)
    return _ns_to_ms(time.perf_counter_ns() - begin)


def _run_phase(
    *,
    runtime: Any,
    command: Any,
    rank: int,
    active_ranks: tuple[int, ...],
) -> list[Any]:
    local_result = None
    if rank in active_ranks:
        runtime._handle_migrate_artifacts(command)
        local_result = runtime.pop_result()

    gathered: list[Any] = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, local_result)
    results = [item for item in gathered if item is not None]
    errors = [item for item in results if getattr(item, "error", None)]
    if errors:
        first = errors[0]
        raise RuntimeError(
            f"runtime_v2 migration phase {command.phase} failed on rank {first.worker_rank}: {first.error}"
        )
    return results


def _schedule_bytes(schedule: Any, sym: Mapping[str, Any]) -> int:
    total = 0
    for edge in schedule.edges:
        if edge.local_copy:
            continue
        dtype = sym["parse_torch_dtype"](edge.src_slice.dtype)
        total += sym["tensor_numel"](tuple(edge.src_slice.shape)) * torch.empty((), dtype=dtype).element_size()
    return total


def _latent_bytes(layout: Any, sym: Mapping[str, Any]) -> int:
    for field in layout.tensors:
        if tuple(field.field_path) != ("latents",):
            continue
        dtype = sym["parse_torch_dtype"](field.dtype)
        return sym["tensor_numel"](tuple(field.global_shape)) * torch.empty((), dtype=dtype).element_size()
    return 0


def _latent_shape(layout: Any) -> str:
    for field in layout.tensors:
        if tuple(field.field_path) == ("latents",):
            return _fmt_shape(tuple(field.global_shape))
    return "<none>"


def _measure_materialize(
    *,
    runtime: Any,
    rank: int,
    dst_group: Any,
    handle: Any,
    case: CaseSpec,
    dist_groups: Mapping[str, Any],
    device: torch.device,
) -> float:
    if rank not in dst_group.ranks:
        return 0.0
    key = runtime._artifact_key(handle.request_id, dst_group.group_id, handle.artifact_id)
    holder = runtime.local_artifacts.get(key)
    if holder is None:
        raise RuntimeError(f"dst artifact missing before materialize: key={key!r}")

    from vllm_omni.diffusion.distributed import parallel_state as omni_parallel_state

    previous_sp = omni_parallel_state._SP
    omni_parallel_state._SP = SimpleNamespace(
        world_size=len(dst_group.ranks),
        device_group=dist_groups[dst_group.group_id],
    )
    begin = time.perf_counter_ns()
    try:
        runtime._materialize_input_artifact(
            artifact=handle,
            value=holder.value,
            group_id=dst_group.group_id,
            request_metadata=case.metadata,
        )
        _sync(device)
        return _ns_to_ms(time.perf_counter_ns() - begin)
    finally:
        omni_parallel_state._SP = previous_sp


def _one_iteration(
    *,
    runtime: Any,
    rank: int,
    transition: str,
    src_group: Any,
    dst_group: Any,
    case: CaseSpec,
    sym: Mapping[str, Any],
    dist_groups: Mapping[str, Any],
    device: torch.device,
    dtype: torch.dtype,
) -> IterationMetrics:
    request_id = f"bench-{case.name}-{transition}"
    handle = sym["ArtifactHandle"](
        request_id=request_id,
        artifact_id="state_denoised_microbench",
        kind=sym["ArtifactKind"].REQUEST_STATE,
        layout=sym["ArtifactLayout"].WORKER_LOCAL,
        codec_id=_codec_id_for_case(case, sym),
    )
    plan = sym["MigrateArtifactsPlan"](
        request_id=request_id,
        src_group_id=src_group.group_id,
        dst_group_id=dst_group.group_id,
        artifacts=(sym["MigrateArtifactSpec"](handle=handle),),
        request_metadata=case.metadata,
    )
    participant_ranks = tuple(sorted(set(src_group.ranks) | set(dst_group.ranks)))

    shard_ms = _seed_source_artifact(
        runtime=runtime,
        rank=rank,
        case=case,
        src_group=src_group,
        handle=handle,
        device=device,
        dtype=dtype,
        sym=sym,
    )
    shard_ms = _max_float(shard_ms, device)

    migrate_id = str(uuid.uuid4())
    command_kwargs = dict(
        migrate_id=migrate_id,
        plan=plan,
        src_group=src_group,
        dst_group=dst_group,
    )

    dist.barrier()
    scheduler_begin = time.perf_counter_ns()
    describe_command = sym["MigrateArtifactsCommand"](
        phase=sym["_MIGRATE_PHASE_DESCRIBE_LAYOUT"],
        **command_kwargs,
    )
    layout_results = _run_phase(
        runtime=runtime,
        command=describe_command,
        rank=rank,
        active_ranks=participant_ranks,
    )
    build_command = sym["MigrateArtifactsCommand"](
        phase=sym["_MIGRATE_PHASE_BUILD_SCHEDULE"],
        layout_results=tuple(layout_results),
        **command_kwargs,
    )
    schedule_results = _run_phase(
        runtime=runtime,
        command=build_command,
        rank=rank,
        active_ranks=(src_group.ranks[0],),
    )
    schedule_result = schedule_results[0]
    schedule = schedule_result.schedule
    if schedule is None:
        raise RuntimeError("runtime_v2 migration build phase returned no schedule")
    scheduler_ms = _ns_to_ms(time.perf_counter_ns() - scheduler_begin)
    scheduler_ms = _max_float(scheduler_ms, device)

    launch_begin = time.perf_counter_ns()
    accept_command = sym["MigrateArtifactsCommand"](
        phase=sym["_MIGRATE_PHASE_ACCEPT_SCHEDULE"],
        schedule=schedule,
        **command_kwargs,
    )
    _run_phase(
        runtime=runtime,
        command=accept_command,
        rank=rank,
        active_ranks=participant_ranks,
    )
    dist.barrier()
    task_launch_ms = _ns_to_ms(time.perf_counter_ns() - launch_begin)
    task_launch_ms = _max_float(task_launch_ms, device)

    execute_begin = time.perf_counter_ns()
    execute_command = sym["MigrateArtifactsCommand"](
        phase=sym["_MIGRATE_PHASE_EXECUTE_TRANSFER"],
        schedule=schedule,
        metadata_payloads=schedule_result.metadata_payloads,
        **command_kwargs,
    )
    _run_phase(
        runtime=runtime,
        command=execute_command,
        rank=rank,
        active_ranks=participant_ranks,
    )
    _sync(device)
    transfer_ms = _ns_to_ms(time.perf_counter_ns() - execute_begin)
    transfer_ms = _max_float(transfer_ms, device)

    materialize_ms = _measure_materialize(
        runtime=runtime,
        rank=rank,
        dst_group=dst_group,
        handle=handle,
        case=case,
        dist_groups=dist_groups,
        device=device,
    )
    materialize_ms = _max_float(materialize_ms, device)

    reference_layout = runtime.codec_registry.describe_output(
        group=dst_group,
        group_rank=0,
        request_metadata=case.metadata,
        artifact=handle,
    )
    return IterationMetrics(
        transition=transition,
        latent_name=case.name,
        latent_shape=_latent_shape(reference_layout),
        bytes=_latent_bytes(reference_layout, sym),
        reshard_bytes=_schedule_bytes(schedule, sym),
        reshard_time_ms=scheduler_ms + task_launch_ms + transfer_ms,
        artifact_materialize_time_ms=materialize_ms,
        artifact_shard_time_ms=shard_ms,
        scheduler_overhead_ms=scheduler_ms,
        task_launch_overhead_ms=task_launch_ms,
        interprocess_transfer_time_ms=transfer_ms,
        group_switch_gap_ms=scheduler_ms + task_launch_ms,
    )


def _summarize(rows: list[IterationMetrics], dit_step_ms: float | None) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, int], list[IterationMetrics]] = {}
    for row in rows:
        grouped.setdefault((row.transition, row.latent_name, row.latent_shape, row.bytes), []).append(row)
    summary = []
    for (transition, name, shape, nbytes), items in sorted(grouped.items()):
        reshard = [item.reshard_time_ms for item in items]
        entry: dict[str, Any] = {
            "transition": transition,
            "latent_name": name,
            "latent_shape": shape,
            "bytes": nbytes,
            "reshard_bytes": items[0].reshard_bytes,
            "p50_ms": _percentile(reshard, 0.50),
            "p95_ms": _percentile(reshard, 0.95),
            "artifact_materialize_time_ms_p50": _percentile(
                [item.artifact_materialize_time_ms for item in items], 0.50
            ),
            "artifact_shard_time_ms_p50": _percentile([item.artifact_shard_time_ms for item in items], 0.50),
            "scheduler_overhead_ms_p50": _percentile([item.scheduler_overhead_ms for item in items], 0.50),
            "task_launch_overhead_ms_p50": _percentile([item.task_launch_overhead_ms for item in items], 0.50),
            "interprocess_transfer_time_ms_p50": _percentile(
                [item.interprocess_transfer_time_ms for item in items], 0.50
            ),
            "group_switch_gap_ms_p50": _percentile([item.group_switch_gap_ms for item in items], 0.50),
        }
        if dit_step_ms is not None and dit_step_ms > 0:
            entry["p95_over_step"] = entry["p95_ms"] / dit_step_ms
            entry["healthy_under_0p1_step"] = entry["p95_over_step"] < 0.1
        summary.append(entry)
    return summary


def _print_summary(summary: list[dict[str, Any]]) -> None:
    print("\nSummary table")
    print("transition\tlatent shape\tbytes\tp50 ms\tp95 ms")
    for row in summary:
        print(
            f"{row['transition']}\t{row['latent_shape']}\t{row['bytes']}\t"
            f"{row['p50_ms']:.3f}\t{row['p95_ms']:.3f}"
        )
    print("\nDetailed p50 metrics")
    detail_cols = (
        "transition",
        "latent_name",
        "latent_shape",
        "bytes",
        "reshard_bytes",
        "artifact_materialize_time_ms_p50",
        "artifact_shard_time_ms_p50",
        "scheduler_overhead_ms_p50",
        "task_launch_overhead_ms_p50",
        "interprocess_transfer_time_ms_p50",
        "group_switch_gap_ms_p50",
    )
    print("\t".join(detail_cols))
    for row in summary:
        print(
            "\t".join(
                f"{row[col]:.3f}" if isinstance(row[col], float) else str(row[col])
                for col in detail_cols
            )
        )
    if summary and "p95_over_step" in summary[0]:
        print("\nHealth check: p95 reshard overhead < 0.1 * one DiT step time")
        for row in summary:
            status = "OK" if row["healthy_under_0p1_step"] else "SLOW"
            print(
                f"{row['transition']} {row['latent_name']}: "
                f"ratio={row['p95_over_step']:.3f} status={status}"
            )


def _write_csv(path: str, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_jsonl(path: str, rows: list[IterationMetrics]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row.__dict__, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtype", default="bfloat16", choices=("float16", "bfloat16", "float32"))
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=30)
    parser.add_argument("--transition", action="append", default=None, help="Transition like SP4->SP2")
    parser.add_argument("--case", action="append", default=None, choices=DEFAULT_CASES)
    parser.add_argument("--csv", default="reshard_microbench_summary.csv")
    parser.add_argument("--jsonl", default="reshard_microbench_raw.jsonl")
    parser.add_argument("--dit-step-ms", type=float, default=None)
    parser.add_argument("--qwen-text-hidden-size", type=int, default=4096)
    parser.add_argument("--qwen-transformer-in-channels", type=int, default=64)
    parser.add_argument("--wan-text-hidden-size", type=int, default=4096)
    parser.add_argument("--wan-latent-channels", type=int, default=16)
    parser.add_argument("--wan-vae-scale-factor-spatial", type=int, default=8)
    parser.add_argument("--wan-vae-scale-factor-temporal", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rank, _world, device = _init_distributed()
    sym = _load_runtime_symbols()
    dtype = getattr(torch, args.dtype)

    group_specs = _build_group_specs(sym)
    dist_groups = _create_dist_groups()
    codec_registry = _build_codec_registry(sym, dtype, args)

    RuntimeBase = sym["MigrationDataPlaneMixin"]

    class RuntimeHarness(RuntimeBase, _ResultCaptureRuntime):
        pass

    runtime = RuntimeHarness(worker_rank=rank, groups=group_specs, codec_registry=codec_registry)

    transitions = tuple(args.transition or DEFAULT_TRANSITIONS)
    parsed_transitions = [(raw, *_parse_transition(raw)) for raw in transitions]
    specs = _case_specs()
    cases = [specs[name] for name in (args.case or DEFAULT_CASES)]

    rows: list[IterationMetrics] = []
    completed = False
    try:
        for transition, src_group_id, dst_group_id in parsed_transitions:
            src_group = group_specs[src_group_id]
            dst_group = group_specs[dst_group_id]
            for case in cases:
                for _ in range(args.warmup):
                    _one_iteration(
                        runtime=runtime,
                        rank=rank,
                        transition=transition,
                        src_group=src_group,
                        dst_group=dst_group,
                        case=case,
                        sym=sym,
                        dist_groups=dist_groups,
                        device=device,
                        dtype=dtype,
                    )
                    dist.barrier()
                for _ in range(args.repetitions):
                    row = _one_iteration(
                        runtime=runtime,
                        rank=rank,
                        transition=transition,
                        src_group=src_group,
                        dst_group=dst_group,
                        case=case,
                        sym=sym,
                        dist_groups=dist_groups,
                        device=device,
                        dtype=dtype,
                    )
                    if rank == 0:
                        rows.append(row)
                    dist.barrier()
        completed = True
    finally:
        if completed and rank == 0:
            summary = _summarize(rows, args.dit_step_ms)
            _print_summary(summary)
            _write_csv(args.csv, summary)
            _write_jsonl(args.jsonl, rows)
            print(f"\nWrote {args.csv} and {args.jsonl}")
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
