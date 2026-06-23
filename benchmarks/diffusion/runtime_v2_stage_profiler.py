#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Per-stage cost-model profiler for runtime_v2 (Wan2.2 video + Qwen Image).

Drives the exact executors used in serving (text_encode, dit_prepare,
timestep_prepare, dit_step_chunk, vae_decode, finalize) over a grid of
(H, W, F) plus text_seq_len, times each task kind with CUDA events synced
across ranks, fits ``t(x) = a*x**2 + b*x + c`` per stage, and writes a
cost-model JSON that SRTF / elastic policies can load. The adapter is
inferred from ``--model`` (override with ``--adapter wan|qwen``).

Wan single-GPU::

    python benchmarks/diffusion/runtime_v2_stage_profiler.py \
        --model Wan-AI/Wan2.2-TI2V-5B-Diffusers \
        --out cost-model/wan_tp1_sp1_cfg1.json

Wan multi-GPU (ulysses=2)::

    torchrun --standalone --nproc_per_node=2 \
        benchmarks/diffusion/runtime_v2_stage_profiler.py \
        --model Wan-AI/Wan2.2-TI2V-5B-Diffusers \
        --ulysses-degree 2 \
        --out cost-model/wan_tp1_sp2_cfg1.json

Qwen Image single-GPU::

    python benchmarks/diffusion/runtime_v2_stage_profiler.py \
        --model Qwen/Qwen-Image --enforce-eager \
        --out cost-model/qwen-image/tp1_ulysses1_ring1_cfg1.json

Qwen Image multi-GPU (ulysses=2)::

    torchrun --standalone --nproc_per_node=2 \
        benchmarks/diffusion/runtime_v2_stage_profiler.py \
        --model Qwen/Qwen-Image --ulysses-degree 2 --enforce-eager \
        --out cost-model/qwen-image/tp1_ulysses2_ring1_cfg1.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist


# Each grid point profiles one stage; per-stage x variable lives in this table.
# kept here (not in adapters) because it expresses the *cost-model* contract,
# not the runtime contract.
STAGE_X_VARS: dict[str, str] = {
    "text_encode": "text_seq_len",
    "dit_prepare": "latent_seq_len",
    "timestep_prepare": "num_steps",
    # Back-compat aggregate stage. For new models this is the CFG curve so old
    # policy configs keep their previous conservative behavior.
    "dit_step_chunk": "latent_seq_len",
    "dit_step_chunk_cfg": "latent_seq_len",
    "dit_step_chunk_nocfg": "latent_seq_len",
    "vae_decode": "voxels",  # H * W * F (output-space pixel-frames)
    "finalize": "ones",      # constant; x is always 1
}


@dataclass(frozen=True)
class GridPoint:
    height: int
    width: int
    num_frames: int


@dataclass
class StageSamples:
    x: list[float] = field(default_factory=list)
    t_ms: list[float] = field(default_factory=list)


def _wan_default_grid() -> list[GridPoint]:
    # Spread that covers small-image (1-frame), short-video, long-video.
    # Latent-token counts range ~[1e3, 4e5] so the quadratic fit has lever arm.
    return [
        GridPoint(480, 480, 17),
        GridPoint(480, 832, 17),
        GridPoint(480, 832, 49),
        GridPoint(480, 832, 81),
        GridPoint(720, 720, 17),
        GridPoint(720, 1280, 17),
        GridPoint(720, 1280, 49),
        GridPoint(720, 1280, 81),
    ]


def _qwen_default_grid() -> list[GridPoint]:
    # Image-only (num_frames=1). Spread covers the serving classes plus
    # interior points so the quadratic fit has lever arm. Qwen latent token
    # count = (H//16)*(W//16): 512->1024, 768->2304, 1024->4096, 1280->6400,
    # 1536->9216 — ~9x range.
    return [
        GridPoint(512, 512, 1),
        GridPoint(768, 768, 1),
        GridPoint(1024, 1024, 1),
        GridPoint(1280, 1280, 1),
        GridPoint(1536, 1536, 1),
    ]


def _parse_grid_arg(raw: str | None) -> list[GridPoint] | None:
    if not raw:
        return None
    spec = json.loads(raw)
    if not isinstance(spec, list):
        raise argparse.ArgumentTypeError("--grid-json must be a JSON list of {height,width,num_frames}")
    return [
        GridPoint(int(item["height"]), int(item["width"]), int(item.get("num_frames", 1)))
        for item in spec
    ]


def _resolve_adapter_kind(args: argparse.Namespace) -> str:
    """Pick the runtime_v2 adapter. ``--adapter auto`` sniffs the model id."""
    override = getattr(args, "adapter", "auto")
    if override and override != "auto":
        return override
    name = str(args.model).lower()
    if "qwen" in name:
        return "qwen"
    if "wan" in name:
        return "wan"
    raise RuntimeError(
        f"could not infer adapter kind from model={args.model!r}; pass --adapter wan|qwen"
    )


def _codec_ids(kind: str) -> tuple[str, str]:
    """(state_codec_id, decoded_codec_id) for the given adapter kind."""
    if kind == "qwen":
        from vllm_omni.diffusion.runtime_v2.adapters.qwen_image import (
            QWEN_IMAGE_DECODED_CODEC_ID,
            QWEN_IMAGE_STATE_CODEC_ID,
        )
        return QWEN_IMAGE_STATE_CODEC_ID, QWEN_IMAGE_DECODED_CODEC_ID
    from vllm_omni.diffusion.runtime_v2.adapters.wan import (
        WAN_DECODED_CODEC_ID,
        WAN_STATE_CODEC_ID,
    )
    return WAN_STATE_CODEC_ID, WAN_DECODED_CODEC_ID


def _latent_seq_len(kind: str, pipeline: Any, point: GridPoint) -> int:
    """Latent token count keyed identically to what the runtime computes, so
    the policy's cost lookup hits the row this point profiled."""
    if kind == "qwen":
        # Mirrors QwenImageRuntimeV2Adapter._estimate_packed_latent_seq_len:
        # packed by VAE spatial scale (8) then transformer patch (2) -> //16.
        packed_h = max(1, int(point.height) // 16)
        packed_w = max(1, int(point.width) // 16)
        return packed_h * packed_w
    from vllm_omni.diffusion.runtime_v2.adapters.wan import WanLatentGeometry
    geom = WanLatentGeometry.from_pipeline(pipeline)
    return geom.dit_latent_seq_len(point.height, point.width, point.num_frames)


def _init_distributed(args: argparse.Namespace) -> tuple[int, int, int, torch.device]:
    if not torch.cuda.is_available():
        raise RuntimeError("stage profiler requires CUDA")

    # When launched without torchrun we still need single-rank distributed
    # because parallel_state.initialize_model_parallel asserts a process group.
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
        raise RuntimeError(f"local_rank={local_rank} exceeds cuda_device_count={torch.cuda.device_count()}")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    expected = args.tensor_parallel_size * args.ulysses_degree * args.ring_degree * args.cfg_parallel_size
    if expected != world:
        raise RuntimeError(
            f"tp*ulysses*ring*cfg = {expected} != world_size {world}; "
            "launch with torchrun --nproc_per_node matching the cartesian product"
        )

    # Defer omni init imports until after device is set — they touch CUDA.
    from vllm_omni.diffusion.distributed.parallel_state import (
        get_sp_group,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm_omni.diffusion.distributed.collective_runtime import (
        init_runtime_v2_collective_runtime,
        register_static_gfc_subgroup,
    )

    init_distributed_environment(world_size=world, rank=rank)
    effective_collective_backend = str(args.collective_backend)
    if world == 1 and effective_collective_backend == "gfc":
        # SP=1 has no cross-rank collective to profile. Initializing GFC here
        # only exercises PyTorch symmetric-memory multicast setup on a
        # singleton process group, which can emit noisy driver warnings and is
        # irrelevant to the measured kernels.
        effective_collective_backend = "torch"
        args.collective_backend = effective_collective_backend
    # Profile under the SAME collective backend serving uses. edf_best_fit
    # serving runs GFC, whose SP all-to-all is faster than torch.distributed;
    # profiling under torch over-estimates every SP>1 DiT timing.
    init_runtime_v2_collective_runtime(
        backend=effective_collective_backend,
        device=device,
        max_collective_bytes=args.gfc_max_collective_mb * 1024 * 1024,
    )
    initialize_model_parallel(
        tensor_parallel_size=args.tensor_parallel_size,
        sequence_parallel_size=args.ulysses_degree * args.ring_degree,
        ulysses_degree=args.ulysses_degree,
        ring_degree=args.ring_degree,
        cfg_parallel_size=args.cfg_parallel_size,
        data_parallel_size=1,
        pipeline_parallel_size=1,
    )
    # initialize_model_parallel builds the SP torch subgroups but does not
    # register them with GFC (only the execution-groups serving path does).
    # Register them so the Ulysses all-to-all routes through GFC — a no-op
    # when --collective-backend=torch.
    if args.ulysses_degree * args.ring_degree > 1:
        sp_coord = get_sp_group()
        for pg in (sp_coord.device_group, sp_coord.ulysses_group, sp_coord.ring_group):
            register_static_gfc_subgroup(pg)
    return rank, world, local_rank, device


def _build_pipeline(args: argparse.Namespace, device: torch.device) -> Any:
    from vllm.config import CompilationConfig, VllmConfig, set_current_vllm_config
    from vllm.v1.worker.workspace import init_workspace_manager

    from vllm.transformers_utils.config import get_hf_file_to_dict

    from vllm_omni.diffusion.data import (
        DiffusionParallelConfig,
        OmniDiffusionConfig,
        TransformerConfig,
    )
    from vllm_omni.diffusion.forward_context import set_forward_context
    from vllm_omni.diffusion.worker.diffusion_model_runner import DiffusionModelRunner

    parallel_config = DiffusionParallelConfig(
        tensor_parallel_size=args.tensor_parallel_size,
        sequence_parallel_size=args.ulysses_degree * args.ring_degree,
        ulysses_degree=args.ulysses_degree,
        ring_degree=args.ring_degree,
        cfg_parallel_size=args.cfg_parallel_size,
        data_parallel_size=1,
        pipeline_parallel_size=1,
    )

    # Sniff diffusers model_index.json the same way omni_diffusion does, so
    # downstream code can locate the WanPipeline class without going through
    # the full Omni stack.
    config_dict = get_hf_file_to_dict("model_index.json", args.model)
    if config_dict is None:
        raise RuntimeError(f"could not find model_index.json under {args.model!r}")
    model_class_name = config_dict.get("_class_name")
    if not model_class_name:
        raise RuntimeError(f"model_index.json missing _class_name for {args.model!r}")

    tf_cfg_dict = get_hf_file_to_dict("transformer/config.json", args.model)
    tf_model_config = TransformerConfig.from_dict(tf_cfg_dict) if tf_cfg_dict else TransformerConfig()

    od_config = OmniDiffusionConfig(
        model=args.model,
        model_class_name=model_class_name,
        tf_model_config=tf_model_config,
        parallel_config=parallel_config,
        boundary_ratio=args.boundary_ratio,
        flow_shift=args.flow_shift,
        enforce_eager=args.enforce_eager,
        enable_runtime_v2=True,
        dtype=torch.bfloat16,
    )

    vllm_config = VllmConfig(compilation_config=CompilationConfig())
    vllm_config.parallel_config.tensor_parallel_size = parallel_config.tensor_parallel_size
    vllm_config.parallel_config.data_parallel_size = parallel_config.data_parallel_size

    init_workspace_manager(device)
    with (
        set_forward_context(vllm_config=vllm_config, omni_diffusion_config=od_config),
        set_current_vllm_config(vllm_config),
    ):
        runner = DiffusionModelRunner(vllm_config=vllm_config, od_config=od_config, device=device)
        runner.load_model(load_format="default")
    pipeline = runner.pipeline
    if pipeline is None:
        raise RuntimeError("pipeline failed to load")
    return pipeline, od_config, vllm_config


def _build_executors(pipeline: Any, kind: str) -> dict[Any, Any]:
    if kind == "qwen":
        from vllm_omni.diffusion.runtime_v2.adapters.qwen_image import QwenImageRuntimeV2Adapter
        return QwenImageRuntimeV2Adapter().build_executors(pipeline)
    from vllm_omni.diffusion.runtime_v2.adapters.wan import WanRuntimeV2Adapter
    return WanRuntimeV2Adapter().build_executors(pipeline)


def _build_request(
    point: GridPoint,
    args: argparse.Namespace,
    kind: str,
    *,
    cfg_enabled: bool = True,
) -> Any:
    from vllm_omni.diffusion.request import OmniDiffusionRequest
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    # +2 inference steps so timed chunks never land on the final-step cleanup branch.
    num_steps = args.warmup_iters + args.bench_iters + 2
    # Image models (Qwen) output PIL by default; video (Wan) outputs np frames.
    decode_output_type = "pil" if kind == "qwen" else "np"
    output_type = "latent" if args.skip_decode_image else decode_output_type

    if kind == "qwen":
        # CFG for Qwen Image == true-CFG with a negative prompt (doubles the
        # transformer batch). do_true_cfg = true_cfg_scale>1 AND negative_prompt
        # is not None (see QwenImageRuntimeV2Adapter.compile_request).
        true_cfg_scale = args.true_cfg_scale if cfg_enabled else 1.0
        negative_prompt = (args.negative_prompt or "blurry, low quality") if cfg_enabled else None
        sampling = OmniDiffusionSamplingParams(
            height=point.height,
            width=point.width,
            num_inference_steps=num_steps,
            true_cfg_scale=true_cfg_scale,
            num_outputs_per_prompt=1,
            max_sequence_length=args.text_seq_len,
            seed=args.seed,
            output_type=output_type,
        )
        return OmniDiffusionRequest(
            prompts=[{"prompt": args.prompt, "negative_prompt": negative_prompt}],
            sampling_params=sampling,
            request_ids=[str(uuid.uuid4())],
        )

    # Wan (video): two-stage CFG via guidance_scale / guidance_scale_2.
    guidance_scale = args.guidance_scale if cfg_enabled else 1.0
    guidance_scale_high = (
        (args.guidance_scale_high if args.guidance_scale_high is not None else args.guidance_scale)
        if cfg_enabled
        else 1.0
    )
    sampling = OmniDiffusionSamplingParams(
        height=point.height,
        width=point.width,
        num_frames=point.num_frames,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
        guidance_scale_2=guidance_scale_high,
        num_outputs_per_prompt=1,
        max_sequence_length=args.text_seq_len,
        seed=args.seed,
        output_type=output_type,
    )
    return OmniDiffusionRequest(
        prompts=[{"prompt": args.prompt, "negative_prompt": args.negative_prompt or None}],
        sampling_params=sampling,
        request_ids=[str(uuid.uuid4())],
    )


def _make_handle(request_id: str, artifact_id: str, *, codec_id: str | None) -> Any:
    from vllm_omni.diffusion.runtime_v2.protocol import ArtifactHandle, ArtifactKind, ArtifactLayout
    return ArtifactHandle(
        request_id=request_id,
        artifact_id=artifact_id,
        kind=ArtifactKind.REQUEST_STATE if codec_id else ArtifactKind.TENSOR,
        layout=ArtifactLayout.WORKER_LOCAL if codec_id else ArtifactLayout.HOST,
        codec_id=codec_id,
    )


def _make_task(
    *,
    request_id: str,
    kind: Any,
    inputs: tuple[Any, ...],
    outputs: tuple[Any, ...],
    step_range: tuple[int, int] | None = None,
    payload: dict[str, Any] | None = None,
) -> Any:
    from vllm_omni.diffusion.runtime_v2.protocol import (
        InferenceTask,
        ParallelSpec,
        StepRange,
        TaskStatus,
    )
    return InferenceTask(
        task_id=f"{request_id}:{kind.value}:0",
        request_id=request_id,
        kind=kind,
        group_id=None,
        parallel_spec=ParallelSpec(),
        status=TaskStatus.READY,
        inputs=inputs,
        outputs=outputs,
        step_range=StepRange(start=step_range[0], end=step_range[1]) if step_range else None,
        payload=payload or {},
    )


def _sync(device: torch.device) -> None:
    torch.cuda.synchronize(device)


def _allreduce_max(value_ms: float, device: torch.device) -> float:
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return value_ms
    t = torch.tensor([float(value_ms)], dtype=torch.float64, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return float(t.item())


def _cuda_time_ms(fn, device: torch.device) -> tuple[float, Any]:
    """Run ``fn()`` once, return (elapsed_ms_max_across_ranks, fn_result)."""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    _sync(device)
    if dist.is_initialized():
        dist.barrier()
    start.record()
    out = fn()
    end.record()
    _sync(device)
    return _allreduce_max(start.elapsed_time(end), device), out


def _voxels(point: GridPoint) -> int:
    return point.height * point.width * point.num_frames


def _profile_dit_step_chunks(
    *,
    request_id: str,
    state_timestep: Any,
    state_denoised: Any,
    state_value: Any,
    args: argparse.Namespace,
    executors: dict[Any, Any],
    device: torch.device,
    samples: dict[str, StageSamples],
    sample_stages: tuple[str, ...],
    latent_seq_len: int,
    run_executor,
) -> Any:
    from vllm_omni.diffusion.runtime_v2.protocol import TaskKind

    dit_iters = args.warmup_iters + args.bench_iters
    inputs_map = {state_timestep.artifact_id: state_value}
    dit_events = [torch.cuda.Event(enable_timing=True) for _ in range(dit_iters + 1)]
    _sync(device)
    if dist.is_initialized():
        dist.barrier()
    dit_events[0].record()
    for step_idx in range(dit_iters):
        task = _make_task(
            request_id=request_id, kind=TaskKind.DIT_STEP_CHUNK,
            inputs=(state_timestep,), outputs=(state_denoised,),
            step_range=(step_idx, step_idx + 1),
        )
        results = run_executor(executors[TaskKind.DIT_STEP_CHUNK], task, inputs_map)
        # chain state forward — same artifact id, mutating value is fine
        state_value = results[0].value
        inputs_map = {state_timestep.artifact_id: state_value}
        dit_events[step_idx + 1].record()
    _sync(device)
    for step_idx in range(args.warmup_iters, dit_iters):
        t_ms = _allreduce_max(
            dit_events[step_idx].elapsed_time(dit_events[step_idx + 1]), device
        )
        for stage in sample_stages:
            samples[stage].x.append(float(latent_seq_len))
            samples[stage].t_ms.append(t_ms)
    return state_value


def _profile_point(
    *,
    point: GridPoint,
    args: argparse.Namespace,
    kind: str,
    executors: dict[Any, Any],
    pipeline: Any,
    device: torch.device,
    samples: dict[str, StageSamples],
    rank: int,
    run_executor,
) -> None:
    from vllm_omni.diffusion.runtime_v2.protocol import TaskKind

    state_codec_id, decoded_codec_id = _codec_ids(kind)

    req = _build_request(point, args, kind)
    request_id = req.request_ids[0]
    state_text = _make_handle(request_id, "state_text", codec_id=state_codec_id)
    state_latent = _make_handle(request_id, "state_latent", codec_id=state_codec_id)
    state_timestep = _make_handle(request_id, "state_timestep", codec_id=state_codec_id)
    state_denoised = _make_handle(request_id, "state_denoised", codec_id=state_codec_id)
    decoded = _make_handle(request_id, "decoded", codec_id=decoded_codec_id)
    output = _make_handle(request_id, "output", codec_id=None)
    req_handle = _make_handle(request_id, "request", codec_id=None)

    # Key the cost model by the SAME latent_seq_len the runtime computes, so
    # the policy's cost lookup hits the row this point profiled. Geometry
    # comes from the served model's own config (see _latent_seq_len).
    lat = _latent_seq_len(kind, pipeline, point)
    vox = _voxels(point)

    if rank == 0:
        print(
            f"  point H={point.height} W={point.width} F={point.num_frames} "
            f"-> latent_seq_len={lat} voxels={vox}",
            flush=True,
        )

    # Single-rank stages (text_encode / vae_decode / finalize) run on one rank
    # and do not parallelize across the SP group — their cost is identical at
    # every SP, so profile them only in the SP=1 run. At SP>1 run text_encode
    # once (the DiT chain needs state_text) and skip vae/finalize entirely.
    # DiT-group stages (dit_prepare / timestep_prepare / dit_step_chunk) run on
    # the SP group and are profiled per SP.
    profile_aux = (int(args.ulysses_degree) * int(args.ring_degree)) == 1
    dit_iters = args.warmup_iters + args.bench_iters
    aux_iters = args.aux_warmup_iters + args.aux_bench_iters

    # ---- text_encode (single-rank) ----
    task = _make_task(
        request_id=request_id, kind=TaskKind.TEXT_ENCODE,
        inputs=(req_handle,), outputs=(state_text,),
    )
    inputs_map: dict[str, Any] = {req_handle.artifact_id: req}
    state_value = None
    for i in range(aux_iters if profile_aux else 1):
        t_ms, results = _cuda_time_ms(
            lambda: run_executor(executors[TaskKind.TEXT_ENCODE], task, inputs_map),
            device,
        )
        state_value = results[0].value
        if profile_aux and i >= args.aux_warmup_iters:
            samples["text_encode"].x.append(float(args.text_seq_len))
            samples["text_encode"].t_ms.append(t_ms)

    # ---- dit_prepare (DiT-group) ----
    task = _make_task(
        request_id=request_id, kind=TaskKind.DIT_PREPARE,
        inputs=(state_text,), outputs=(state_latent,),
    )
    inputs_map = {state_text.artifact_id: state_value}
    for i in range(dit_iters):
        t_ms, results = _cuda_time_ms(
            lambda: run_executor(executors[TaskKind.DIT_PREPARE], task, inputs_map),
            device,
        )
        state_value = results[0].value
        if i >= args.warmup_iters:
            samples["dit_prepare"].x.append(float(lat))
            samples["dit_prepare"].t_ms.append(t_ms)

    # ---- timestep_prepare (DiT-group) ----
    task = _make_task(
        request_id=request_id, kind=TaskKind.TIMESTEP_PREPARE,
        inputs=(state_latent,), outputs=(state_timestep,),
    )
    inputs_map = {state_latent.artifact_id: state_value}
    for i in range(dit_iters):
        t_ms, results = _cuda_time_ms(
            lambda: run_executor(executors[TaskKind.TIMESTEP_PREPARE], task, inputs_map),
            device,
        )
        state_value = results[0].value
        if i >= args.warmup_iters:
            samples["timestep_prepare"].x.append(float(req.sampling_params.num_inference_steps))
            samples["timestep_prepare"].t_ms.append(t_ms)

    # ---- dit_step_chunk_cfg (DiT-group): pipelined timing ----
    # Run all chunks back-to-back with NO per-chunk sync, recording one CUDA
    # event after each. The interval between consecutive events is the
    # steady-state per-chunk GPU time a loaded server actually achieves: its
    # CPU prep for chunk N+1 overlaps chunk N's GPU kernels. A per-chunk sync
    # (the old _cuda_time_ms path) instead measures isolated latency with the
    # CPU-prep gap included, which over-states cost ~2x — verified against
    # live-serving exec_gpu_ms. Natural CUDA launch backpressure bounds the
    # in-flight depth, so this does not blow memory.
    state_value = _profile_dit_step_chunks(
        request_id=request_id,
        state_timestep=state_timestep,
        state_denoised=state_denoised,
        state_value=state_value,
        args=args,
        executors=executors,
        device=device,
        samples=samples,
        sample_stages=("dit_step_chunk", "dit_step_chunk_cfg"),
        latent_seq_len=lat,
        run_executor=run_executor,
    )

    # ---- dit_step_chunk_nocfg (DiT-group): same shape, guidance disabled ----
    # Build a separate request so the text executor creates a no-CFG state.
    # Prep stages are only used to construct the state; their costs were
    # already sampled above and are not duplicated in the model.
    req_nocfg = _build_request(point, args, kind, cfg_enabled=False)
    request_id_nocfg = req_nocfg.request_ids[0]
    state_text_nocfg = _make_handle(request_id_nocfg, "state_text", codec_id=state_codec_id)
    state_latent_nocfg = _make_handle(request_id_nocfg, "state_latent", codec_id=state_codec_id)
    state_timestep_nocfg = _make_handle(request_id_nocfg, "state_timestep", codec_id=state_codec_id)
    state_denoised_nocfg = _make_handle(request_id_nocfg, "state_denoised", codec_id=state_codec_id)
    req_handle_nocfg = _make_handle(request_id_nocfg, "request", codec_id=None)

    task = _make_task(
        request_id=request_id_nocfg, kind=TaskKind.TEXT_ENCODE,
        inputs=(req_handle_nocfg,), outputs=(state_text_nocfg,),
    )
    _, results = _cuda_time_ms(
        lambda: run_executor(
            executors[TaskKind.TEXT_ENCODE],
            task,
            {req_handle_nocfg.artifact_id: req_nocfg},
        ),
        device,
    )
    state_value_nocfg = results[0].value

    task = _make_task(
        request_id=request_id_nocfg, kind=TaskKind.DIT_PREPARE,
        inputs=(state_text_nocfg,), outputs=(state_latent_nocfg,),
    )
    _, results = _cuda_time_ms(
        lambda: run_executor(
            executors[TaskKind.DIT_PREPARE],
            task,
            {state_text_nocfg.artifact_id: state_value_nocfg},
        ),
        device,
    )
    state_value_nocfg = results[0].value

    task = _make_task(
        request_id=request_id_nocfg, kind=TaskKind.TIMESTEP_PREPARE,
        inputs=(state_latent_nocfg,), outputs=(state_timestep_nocfg,),
    )
    _, results = _cuda_time_ms(
        lambda: run_executor(
            executors[TaskKind.TIMESTEP_PREPARE],
            task,
            {state_latent_nocfg.artifact_id: state_value_nocfg},
        ),
        device,
    )
    state_value_nocfg = results[0].value
    _profile_dit_step_chunks(
        request_id=request_id_nocfg,
        state_timestep=state_timestep_nocfg,
        state_denoised=state_denoised_nocfg,
        state_value=state_value_nocfg,
        args=args,
        executors=executors,
        device=device,
        samples=samples,
        sample_stages=("dit_step_chunk_nocfg",),
        latent_seq_len=lat,
        run_executor=run_executor,
    )

    # ---- vae_decode + finalize (single-rank) ----
    # Profiled only at SP=1; at SP>1 these are unchanged, so skip them.
    if profile_aux:
        if not args.skip_decode_image:
            # The output_type override forces VAE path; switch back.
            state_value.output_type = "np"
        task = _make_task(
            request_id=request_id, kind=TaskKind.VAE_DECODE,
            inputs=(state_denoised,), outputs=(decoded,),
        )
        inputs_map = {state_denoised.artifact_id: state_value}
        decoded_value = None
        for i in range(aux_iters):
            t_ms, results = _cuda_time_ms(
                lambda: run_executor(executors[TaskKind.VAE_DECODE], task, inputs_map),
                device,
            )
            decoded_value = results[0].value
            if i >= args.aux_warmup_iters:
                samples["vae_decode"].x.append(float(vox))
                samples["vae_decode"].t_ms.append(t_ms)

        task = _make_task(
            request_id=request_id, kind=TaskKind.FINALIZE,
            inputs=(decoded,), outputs=(output,),
        )
        inputs_map = {decoded.artifact_id: decoded_value}
        for i in range(aux_iters):
            t_ms, _ = _cuda_time_ms(
                lambda: run_executor(executors[TaskKind.FINALIZE], task, inputs_map),
                device,
            )
            if i >= args.aux_warmup_iters:
                samples["finalize"].x.append(1.0)
                samples["finalize"].t_ms.append(t_ms)


def _fit_stage(samples: StageSamples) -> dict[str, Any]:
    if len(samples.x) < 2:
        return {
            "coeffs": {"a": 0.0, "b": 0.0, "c": float(np.mean(samples.t_ms)) if samples.t_ms else 0.0},
            "r2": float("nan"),
            "samples": [[x, t] for x, t in zip(samples.x, samples.t_ms)],
        }
    x = np.asarray(samples.x, dtype=np.float64)
    y = np.asarray(samples.t_ms, dtype=np.float64)
    deg = 2 if len(samples.x) >= 3 and np.ptp(x) > 0 else 1
    coeffs = np.polyfit(x, y, deg=deg)
    a, b, c = (coeffs[0], coeffs[1], coeffs[2]) if deg == 2 else (0.0, coeffs[0], coeffs[1])
    pred = a * x * x + b * x + c
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return {
        "coeffs": {"a": float(a), "b": float(b), "c": float(c)},
        "r2": float(r2),
        "samples": [[float(xi), float(ti)] for xi, ti in zip(samples.x, samples.t_ms)],
    }


def _write_output(args: argparse.Namespace, kind: str, samples: dict[str, StageSamples]) -> None:
    options: dict[str, Any] = {
        "adapter": kind,
        "enforce_eager": args.enforce_eager,
        "collective_backend": args.collective_backend,
        "text_seq_len": args.text_seq_len,
        "profile_nocfg": True,
        "warmup_iters": args.warmup_iters,
        "bench_iters": args.bench_iters,
    }
    if kind == "qwen":
        options["true_cfg_scale"] = args.true_cfg_scale
    else:
        options["boundary_ratio"] = args.boundary_ratio
        options["flow_shift"] = args.flow_shift
        options["guidance_scale"] = args.guidance_scale
        options["guidance_scale_high"] = args.guidance_scale_high
    out: dict[str, Any] = {
        "model": args.model,
        "parallelism": {
            "tp": args.tensor_parallel_size,
            "sp": args.ulysses_degree * args.ring_degree,
            "ulysses": args.ulysses_degree,
            "ring": args.ring_degree,
            "cfg": args.cfg_parallel_size,
        },
        "options": options,
        "stages": {},
    }
    for stage, x_name in STAGE_X_VARS.items():
        # SP>1 runs skip single-rank stages — omit empty stages from the JSON.
        if not samples[stage].x:
            continue
        out["stages"][stage] = {"x_name": x_name, **_fit_stage(samples[stage])}
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, sort_keys=True))
    print(f"\nwrote cost-model to {out_path}", flush=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="Wan-AI/Wan2.2-TI2V-5B-Diffusers")
    parser.add_argument("--adapter", choices=("auto", "wan", "qwen"), default="auto",
                        help="runtime_v2 adapter to profile; 'auto' infers from --model")
    parser.add_argument("--out", required=True, help="output JSON path for cost-model coefficients")
    parser.add_argument("--grid-json", default=None,
                        help="optional JSON list of {height,width,num_frames} points (num_frames optional, defaults to 1)")

    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--ulysses-degree", type=int, default=1)
    parser.add_argument("--ring-degree", type=int, default=1)
    parser.add_argument("--cfg-parallel-size", type=int, default=1)

    parser.add_argument("--boundary-ratio", type=float, default=0.875)
    parser.add_argument("--flow-shift", type=float, default=5.0)
    parser.add_argument("--enforce-eager", action="store_true",
                        help="disable torch.compile — recommended for sweeps with many shapes")
    parser.add_argument("--guidance-scale", type=float, default=4.0)
    parser.add_argument("--guidance-scale-high", type=float, default=None)
    parser.add_argument("--true-cfg-scale", type=float, default=4.0,
                        help="Qwen Image true-CFG scale for the dit_step_chunk_cfg curve (>1 enables CFG)")

    parser.add_argument("--prompt", default="a serene lakeside sunrise with mist over the water")
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--text-seq-len", type=int, default=512)
    parser.add_argument("--skip-decode-image", action="store_true",
                        help="leave output_type=latent and don't time vae_decode (still runs once for chaining)")

    parser.add_argument("--warmup-iters", type=int, default=5,
                        help="discarded warmup iterations for DiT-group stages "
                             "(a few iters settle cuBLAS/allocator first-call cost)")
    parser.add_argument("--bench-iters", type=int, default=20,
                        help="timed iterations for DiT-group stages that feed the fit")
    parser.add_argument("--aux-warmup-iters", type=int, default=2,
                        help="discarded warmup iterations for single-rank stages "
                             "(text_encode/vae_decode/finalize)")
    parser.add_argument("--aux-bench-iters", type=int, default=5,
                        help="timed iterations for single-rank stages (profiled at SP=1 only)")
    parser.add_argument("--collective-backend", choices=("torch", "gfc"), default="torch",
                        help="SP collective backend to profile under; use 'gfc' to match "
                             "edf_best_fit serving (its SP all-to-all is faster than torch)")
    parser.add_argument("--gfc-max-collective-mb", type=int, default=1024,
                        help="GFC symmetric-memory budget per rank in MiB (--collective-backend gfc)")
    parser.add_argument("--master-port", type=int, default=29500)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    kind = _resolve_adapter_kind(args)
    if kind == "qwen":
        if args.true_cfg_scale <= 1.0:
            raise ValueError(
                "stage profiler writes dit_step_chunk_cfg from --true-cfg-scale, which must "
                "be > 1.0. The profiler always measures dit_step_chunk_nocfg separately with "
                "true_cfg_scale=1.0."
            )
    else:
        effective_guidance_high = (
            args.guidance_scale if args.guidance_scale_high is None else args.guidance_scale_high
        )
        if args.guidance_scale <= 1.0 or effective_guidance_high <= 1.0:
            raise ValueError(
                "stage profiler writes dit_step_chunk_cfg from --guidance-scale/--guidance-scale-high; "
                "both must be > 1.0. The profiler always measures dit_step_chunk_nocfg "
                "separately with guidance_scale=1.0."
            )
    rank, world, local_rank, device = _init_distributed(args)
    if rank == 0:
        print(
            f"profiler start: model={args.model} world={world} "
            f"tp={args.tensor_parallel_size} ulysses={args.ulysses_degree} "
            f"ring={args.ring_degree} cfg={args.cfg_parallel_size} eager={args.enforce_eager}",
            flush=True,
        )

    pipeline, od_config, vllm_config = _build_pipeline(args, device)
    executors = _build_executors(pipeline, kind)

    default_grid = _qwen_default_grid() if kind == "qwen" else _wan_default_grid()
    grid = _parse_grid_arg(args.grid_json) or default_grid
    samples: dict[str, StageSamples] = {stage: StageSamples() for stage in STAGE_X_VARS}

    # Wan2.2 transformer.forward reads omni_diffusion_config from the
    # forward-context contextvar, and the _sp_plan hooks mutate
    # sp_original_seq_len / sp_padding_size on that same ctx. Serving opens a
    # FRESH set_forward_context per task (multiproc_worker.py:1020); if we
    # share one ctx across the grid, stale sp_* fields make the next call's
    # hidden_states_mask shape mismatch the new query length. Mirror serving
    # by re-entering the context manager for every executor.execute call.
    from vllm.config import set_current_vllm_config
    from vllm_omni.diffusion.forward_context import set_forward_context

    def run_executor(executor: Any, task: Any, resolved_inputs: dict[str, Any]):
        with set_forward_context(vllm_config=vllm_config, omni_diffusion_config=od_config):
            return executor.execute(task, resolved_inputs)

    with set_current_vllm_config(vllm_config):
        for i, point in enumerate(grid):
            if rank == 0:
                print(
                    f"\n[{i + 1}/{len(grid)}] profiling H={point.height} W={point.width} F={point.num_frames}",
                    flush=True,
                )
            with torch.inference_mode():
                _profile_point(
                    point=point,
                    args=args,
                    kind=kind,
                    executors=executors,
                    pipeline=pipeline,
                    device=device,
                    samples=samples,
                    rank=rank,
                    run_executor=run_executor,
                )
            torch.cuda.empty_cache()

    if rank == 0:
        _write_output(args, kind, samples)

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
