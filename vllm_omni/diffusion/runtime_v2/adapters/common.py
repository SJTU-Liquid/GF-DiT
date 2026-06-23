# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from vllm_omni.diffusion.runtime_v2.protocol import (
    ArtifactHandle,
    ArtifactKind,
    ArtifactLayout,
    ArtifactValue,
    InferenceTask,
    OutputArtifactLayout,
    ParallelSpec,
    RequestExecutionPlan,
    RESHARD_PAYLOAD_DST_GROUP_ID,
    RESHARD_PAYLOAD_SRC_GROUP_ID,
    StepRange,
    TaskKind,
    TaskStatus,
    TensorFieldLayout,
    TensorLayout,
    TensorLayoutKind,
    TensorShardSpec,
)

STANDARD_TASK_KINDS: tuple[TaskKind, ...] = (
    TaskKind.TEXT_ENCODE,
    TaskKind.DIT_PREPARE,
    TaskKind.TIMESTEP_PREPARE,
    TaskKind.DIT_STEP_CHUNK,
    TaskKind.VAE_DECODE,
    TaskKind.FINALIZE,
)

# Stage labels accepted in `stage_group_ids` for build_chunked_dit_plan().
STAGE_AUX = "aux"
STAGE_DIT = "dit"

# TaskKind -> stage label mapping. The aux stage runs text encoding, VAE decode
# and the host-side finalize task; the dit stage runs DiT preparation and the
# denoising loop.
_TASK_KIND_TO_STAGE: dict[TaskKind, str] = {
    TaskKind.TEXT_ENCODE: STAGE_AUX,
    TaskKind.VAE_DECODE: STAGE_AUX,
    TaskKind.FINALIZE: STAGE_AUX,
    TaskKind.DIT_PREPARE: STAGE_DIT,
    TaskKind.TIMESTEP_PREPARE: STAGE_DIT,
    TaskKind.DIT_STEP_CHUNK: STAGE_DIT,
}


REPLICATED_TENSOR_LAYOUT = TensorLayout(kind=TensorLayoutKind.REPLICATED)


def make_request_state_handle(
    request_id: str,
    artifact_id: str,
    producer_suffix: str,
    *,
    codec_id: str,
) -> ArtifactHandle:
    return ArtifactHandle(
        request_id=request_id,
        artifact_id=artifact_id,
        kind=ArtifactKind.REQUEST_STATE,
        layout=ArtifactLayout.WORKER_LOCAL,
        producer_task_id=f"{request_id}:{producer_suffix}:0",
        codec_id=codec_id,
    )


def dtype_name(dtype: Any) -> str:
    return str(dtype)


def replicated_tensor_field(
    field_path: str | tuple[str, ...],
    local_shape: tuple[int, ...],
    dtype: Any,
) -> TensorFieldLayout:
    if isinstance(field_path, str):
        field_path = (field_path,)
    return TensorFieldLayout(
        field_path=field_path,
        local_shape=local_shape,
        dtype=dtype_name(dtype),
        layout=REPLICATED_TENSOR_LAYOUT,
        global_shape=local_shape,
        global_offset=tuple(0 for _ in local_shape),
    )


def _balanced_shard_bounds(size: int, world_size: int, rank: int) -> tuple[int, int]:
    if size < 0:
        raise ValueError(f"sharded dimension size must be >= 0, got {size}")
    if world_size < 1:
        raise ValueError(f"shard world_size must be >= 1, got {world_size}")
    if rank < 0 or rank >= world_size:
        raise ValueError(f"shard rank {rank} out of range for world_size={world_size}")
    base = size // world_size
    remainder = size % world_size
    start = rank * base + min(rank, remainder)
    length = base + (1 if rank < remainder else 0)
    return start, length


def sharded_tensor_field(
    field_path: str | tuple[str, ...],
    global_shape: tuple[int, ...],
    dtype: Any,
    *,
    shard_dim: int,
    shard_world_size: int,
    shard_rank: int,
) -> TensorFieldLayout:
    if isinstance(field_path, str):
        field_path = (field_path,)
    global_shape = tuple(int(dim) for dim in global_shape)
    if shard_dim < 0:
        shard_dim += len(global_shape)
    if shard_dim < 0 or shard_dim >= len(global_shape):
        raise ValueError(f"shard_dim {shard_dim} out of range for global_shape={global_shape}")
    offset, local = _balanced_shard_bounds(global_shape[shard_dim], shard_world_size, shard_rank)
    local_shape = list(global_shape)
    local_shape[shard_dim] = local
    global_offset = [0 for _ in global_shape]
    global_offset[shard_dim] = offset
    return TensorFieldLayout(
        field_path=field_path,
        local_shape=tuple(local_shape),
        dtype=dtype_name(dtype),
        layout=TensorLayout(kind=TensorLayoutKind.SHARDED, shard=TensorShardSpec(dim=shard_dim)),
        global_shape=global_shape,
        global_offset=tuple(global_offset),
    )


def sp_sharded_tensor_field(
    group: Any,
    group_rank: int,
    field_path: str | tuple[str, ...],
    global_shape: tuple[int, ...],
    dtype: Any,
    *,
    shard_dim: int,
) -> TensorFieldLayout:
    sp_world_size = int(getattr(group.parallel_spec, "sp", 1) or 1)
    if sp_world_size <= 1:
        return replicated_tensor_field(field_path, global_shape, dtype)
    # Runtime v2 rank generation orders ranks as tp-sp-pp-cfg-dp. Dynamic
    # step switching only supports tp=1, pp=1, dp=1, so group_rank % sp is the
    # SP rank while CFG ranks replicate the same SP shard.
    sp_rank = int(group_rank) % sp_world_size
    return sharded_tensor_field(
        field_path,
        global_shape,
        dtype,
        shard_dim=shard_dim,
        shard_world_size=sp_world_size,
        shard_rank=sp_rank,
    )


def output_artifact_layout(handle: ArtifactHandle, tensors: tuple[TensorFieldLayout, ...]) -> OutputArtifactLayout:
    return OutputArtifactLayout(handle=handle, tensors=tensors)


def effective_chunk_seq_len(latent_seq_len: int, chunk_steps: int) -> int:
    steps = max(1, int(chunk_steps))
    effective = float(max(1, latent_seq_len)) * math.sqrt(float(steps))
    return max(1, int(round(effective)))


def _normalize_dit_step_schedule(
    *,
    num_steps: int,
    default_group_id: str | None,
    dit_step_schedule: tuple[Mapping[str, Any], ...] | None,
) -> tuple[tuple[int, int, str | None], ...]:
    if not dit_step_schedule:
        # In the legacy fcfs/srtf path tasks may intentionally be unpinned and
        # let the scheduler choose an execution group from the topology.
        return ((0, num_steps, default_group_id),)

    normalized: list[tuple[int, int, str | None]] = []
    expected_start = 0
    for index, item in enumerate(dit_step_schedule):
        start = int(item.get("start", 0))
        end_raw = item.get("end")
        end = num_steps if end_raw is None else min(int(end_raw), num_steps)
        group_id = str(item.get("group_id", ""))
        if not group_id:
            raise ValueError(f"dit_step_schedule[{index}] requires non-empty group_id")
        if start != expected_start:
            raise ValueError(
                "dit_step_schedule must cover steps contiguously from 0: "
                f"entry {index} starts at {start}, expected {expected_start}"
            )
        if end <= start:
            raise ValueError(f"dit_step_schedule[{index}] has empty range [{start}, {end})")
        normalized.append((start, end, group_id))
        expected_start = end
        if expected_start >= num_steps:
            break
    if expected_start != num_steps:
        raise ValueError(
            f"dit_step_schedule must cover all {num_steps} steps; covered only [0, {expected_start})"
        )
    return tuple(normalized)


def _group_for_step(step: int, schedule: tuple[tuple[int, int, str | None], ...]) -> tuple[int, str | None]:
    for _start, end, group_id in schedule:
        if step < end:
            return end, group_id
    raise ValueError(f"no DiT group configured for step {step}")


def build_chunked_dit_plan(
    *,
    request_id: str,
    request_value: Any,
    request_type: str,
    num_steps: int,
    chunk_size: int,
    priority: int,
    group_id: str | None,
    text_seq_len: int,
    latent_seq_len: int,
    state_codec_id: str,
    decoded_codec_id: str | None,
    metadata: Mapping[str, Any] | None = None,
    stage_group_ids: Mapping[str, str] | None = None,
    dit_step_schedule: tuple[Mapping[str, Any], ...] | None = None,
    dit_step_do_true_cfg: tuple[bool, ...] | None = None,
) -> RequestExecutionPlan:
    if num_steps < 1:
        raise ValueError(f"num_steps must be >= 1, got {num_steps}")
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
    if text_seq_len < 1:
        raise ValueError(f"text_seq_len must be >= 1, got {text_seq_len}")
    if latent_seq_len < 1:
        raise ValueError(f"latent_seq_len must be >= 1, got {latent_seq_len}")
    if dit_step_do_true_cfg is not None and len(dit_step_do_true_cfg) != num_steps:
        raise ValueError(
            f"dit_step_do_true_cfg length must match num_steps={num_steps}; "
            f"got {len(dit_step_do_true_cfg)}"
        )
    # When stage_group_ids is provided, override per-task group_id by stage and
    # insert RESHARD tasks at the encode->dit and dit->vae boundaries. Default
    # behavior (parameter omitted) is unchanged: every task takes the single
    # `group_id` argument and no RESHARD tasks are emitted.
    aux_group_id: str | None = None
    dit_group_id: str | None = None
    if stage_group_ids is not None:
        try:
            aux_group_id = stage_group_ids[STAGE_AUX]
            dit_group_id = stage_group_ids[STAGE_DIT]
        except KeyError as exc:
            raise ValueError(
                f"stage_group_ids must contain both {STAGE_AUX!r} and {STAGE_DIT!r} keys; got {dict(stage_group_ids)!r}"
            ) from exc
        if not aux_group_id or not dit_group_id:
            raise ValueError(
                f"stage_group_ids values must be non-empty group_ids; got {dict(stage_group_ids)!r}"
            )
        if aux_group_id == dit_group_id:
            raise ValueError(
                f"stage_group_ids requires distinct aux/dit groups; got both = {aux_group_id!r}"
            )

    def _group_for_kind(kind: TaskKind) -> str | None:
        if stage_group_ids is None:
            return group_id
        stage = _TASK_KIND_TO_STAGE.get(kind)
        if stage is None:
            raise ValueError(f"no stage mapping for task kind {kind!r}")
        return aux_group_id if stage == STAGE_AUX else dit_group_id

    dit_schedule = _normalize_dit_step_schedule(
        num_steps=num_steps,
        default_group_id=_group_for_kind(TaskKind.DIT_STEP_CHUNK),
        dit_step_schedule=dit_step_schedule,
    )
    first_dit_group_id = dit_schedule[0][2]
    last_dit_group_id = dit_schedule[-1][2]

    req_handle = ArtifactHandle(
        request_id=request_id,
        artifact_id="request",
        kind=ArtifactKind.REQUEST_STATE,
        layout=ArtifactLayout.HOST,
    )
    state_text = make_request_state_handle(request_id, "state_text", "text_encode", codec_id=state_codec_id)
    state_latent = make_request_state_handle(request_id, "state_latent", "dit_prepare", codec_id=state_codec_id)
    state_timestep = make_request_state_handle(
        request_id,
        "state_timestep",
        "timestep_prepare",
        codec_id=state_codec_id,
    )
    state_denoised = make_request_state_handle(
        request_id,
        "state_denoised",
        "dit_step_chunk:final",
        codec_id=state_codec_id,
    )
    decoded = ArtifactHandle(
        request_id=request_id,
        artifact_id="decoded",
        kind=ArtifactKind.TENSOR,
        layout=ArtifactLayout.HOST,
        producer_task_id=f"{request_id}:vae_decode:0",
        codec_id=decoded_codec_id,
    )
    final_output = ArtifactHandle(
        request_id=request_id,
        artifact_id="output",
        kind=ArtifactKind.OUTPUT,
        layout=ArtifactLayout.HOST,
        producer_task_id=f"{request_id}:finalize:0",
    )

    tasks: dict[str, InferenceTask] = {
        f"{request_id}:text_encode:0": InferenceTask(
            task_id=f"{request_id}:text_encode:0",
            request_id=request_id,
            kind=TaskKind.TEXT_ENCODE,
            group_id=_group_for_kind(TaskKind.TEXT_ENCODE),
            parallel_spec=ParallelSpec(),
            status=TaskStatus.PENDING,
            priority=priority,
            inputs=(req_handle,),
            outputs=(state_text,),
            payload={"seq_len": text_seq_len, "request_stage": TaskKind.TEXT_ENCODE.value},
        ),
        f"{request_id}:dit_prepare:0": InferenceTask(
            task_id=f"{request_id}:dit_prepare:0",
            request_id=request_id,
            kind=TaskKind.DIT_PREPARE,
            group_id=first_dit_group_id,
            parallel_spec=ParallelSpec(),
            status=TaskStatus.PENDING,
            priority=priority,
            dependencies=(f"{request_id}:text_encode:0",),
            inputs=(state_text,),
            outputs=(state_latent,),
            payload={"seq_len": latent_seq_len, "request_stage": TaskKind.DIT_PREPARE.value},
        ),
        f"{request_id}:timestep_prepare:0": InferenceTask(
            task_id=f"{request_id}:timestep_prepare:0",
            request_id=request_id,
            kind=TaskKind.TIMESTEP_PREPARE,
            group_id=first_dit_group_id,
            parallel_spec=ParallelSpec(),
            status=TaskStatus.PENDING,
            priority=priority,
            dependencies=(f"{request_id}:dit_prepare:0",),
            inputs=(state_latent,),
            outputs=(state_timestep,),
            payload={"seq_len": latent_seq_len, "request_stage": TaskKind.TIMESTEP_PREPARE.value},
        ),
        f"{request_id}:vae_decode:0": InferenceTask(
            task_id=f"{request_id}:vae_decode:0",
            request_id=request_id,
            kind=TaskKind.VAE_DECODE,
            group_id=_group_for_kind(TaskKind.VAE_DECODE),
            parallel_spec=ParallelSpec(),
            status=TaskStatus.PENDING,
            priority=priority,
            payload={"seq_len": latent_seq_len, "request_stage": TaskKind.VAE_DECODE.value},
            outputs=(decoded,),
        ),
        f"{request_id}:finalize:0": InferenceTask(
            task_id=f"{request_id}:finalize:0",
            request_id=request_id,
            kind=TaskKind.FINALIZE,
            group_id=_group_for_kind(TaskKind.FINALIZE),
            parallel_spec=ParallelSpec(),
            status=TaskStatus.PENDING,
            priority=priority,
            dependencies=(f"{request_id}:vae_decode:0",),
            inputs=(decoded,),
            outputs=(final_output,),
            payload={"seq_len": 1, "request_stage": TaskKind.FINALIZE.value},
        ),
    }

    denoise_ids: list[str] = []
    prev_task_id = f"{request_id}:timestep_prepare:0"
    prev_state = state_timestep
    prev_group_id = first_dit_group_id
    idx = 0
    start = 0
    while start < num_steps:
        schedule_end, current_group_id = _group_for_step(start, dit_schedule)
        end = min(start + chunk_size, schedule_end, num_steps)
        is_last = end == num_steps
        if prev_group_id != current_group_id:
            migrated_state = make_request_state_handle(
                request_id,
                f"{prev_state.artifact_id}_to_{current_group_id}_{idx}",
                f"reshard_dit_step:{idx}",
                codec_id=state_codec_id,
            )
            reshard_task_id = f"{request_id}:reshard_dit_step:{idx}"
            tasks[reshard_task_id] = InferenceTask(
                task_id=reshard_task_id,
                request_id=request_id,
                kind=TaskKind.RESHARD,
                group_id=None,
                parallel_spec=ParallelSpec(),
                status=TaskStatus.PENDING,
                priority=priority,
                dependencies=(prev_task_id,),
                inputs=(prev_state,),
                outputs=(migrated_state,),
                payload={
                    RESHARD_PAYLOAD_SRC_GROUP_ID: prev_group_id,
                    RESHARD_PAYLOAD_DST_GROUP_ID: current_group_id,
                    "request_stage": "reshard",
                },
            )
            prev_task_id = reshard_task_id
            prev_state = migrated_state
            prev_group_id = current_group_id
        task_id = f"{request_id}:dit_step_chunk:{idx}"
        chunk_do_true_cfg = (
            any(dit_step_do_true_cfg[start:end])
            if dit_step_do_true_cfg is not None
            else None
        )
        cost_model_stage = None
        if chunk_do_true_cfg is not None:
            cost_model_stage = (
                f"{TaskKind.DIT_STEP_CHUNK.value}_cfg"
                if chunk_do_true_cfg
                else f"{TaskKind.DIT_STEP_CHUNK.value}_nocfg"
            )
        out_state = (
            state_denoised
            if is_last
            else make_request_state_handle(
                request_id,
                f"state_denoised_{idx}",
                f"dit_step_chunk:{idx}",
                codec_id=state_codec_id,
            )
        )
        tasks[task_id] = InferenceTask(
            task_id=task_id,
            request_id=request_id,
            kind=TaskKind.DIT_STEP_CHUNK,
            group_id=current_group_id,
            parallel_spec=ParallelSpec(),
            status=TaskStatus.PENDING,
            priority=priority,
            dependencies=(prev_task_id,),
            inputs=(prev_state,),
            outputs=(out_state,),
            step_range=StepRange(start=start, end=end),
            payload={
                "seq_len": effective_chunk_seq_len(latent_seq_len, end - start),
                "request_stage": TaskKind.DIT_STEP_CHUNK.value,
                **({} if chunk_do_true_cfg is None else {"do_true_cfg": chunk_do_true_cfg}),
                **({} if cost_model_stage is None else {"cost_model_stage": cost_model_stage}),
            },
        )
        denoise_ids.append(task_id)
        prev_task_id = task_id
        prev_state = out_state
        prev_group_id = current_group_id
        idx += 1
        start = end

    tasks[f"{request_id}:vae_decode:0"].dependencies = (denoise_ids[-1],)
    tasks[f"{request_id}:vae_decode:0"].inputs = (state_denoised,)

    if stage_group_ids is not None:
        # Insert RESHARD tasks at the encode->dit and dit->vae boundaries and
        # rewrite the downstream consumers to read the migrated artifact.
        # The RESHARD output handles use distinct artifact_ids to keep the
        # global artifact_store keyspace unambiguous and to make the
        # WorkerLocalArtifactRef.group_id swap explicit at dispatch time.
        text_to_dit_handle = ArtifactHandle(
            request_id=request_id,
            artifact_id="state_text_dit",
            kind=ArtifactKind.REQUEST_STATE,
            layout=ArtifactLayout.WORKER_LOCAL,
            producer_task_id=f"{request_id}:reshard_text_to_dit:0",
            codec_id=state_codec_id,
        )
        text_to_dit_task_id = f"{request_id}:reshard_text_to_dit:0"
        tasks[text_to_dit_task_id] = InferenceTask(
            task_id=text_to_dit_task_id,
            request_id=request_id,
            kind=TaskKind.RESHARD,
            group_id=None,
            parallel_spec=ParallelSpec(),
            status=TaskStatus.PENDING,
            priority=priority,
            dependencies=(f"{request_id}:text_encode:0",),
            inputs=(state_text,),
            outputs=(text_to_dit_handle,),
            payload={
                RESHARD_PAYLOAD_SRC_GROUP_ID: aux_group_id,
                RESHARD_PAYLOAD_DST_GROUP_ID: first_dit_group_id,
                "request_stage": "reshard",
            },
        )
        dit_prepare = tasks[f"{request_id}:dit_prepare:0"]
        dit_prepare.dependencies = (text_to_dit_task_id,)
        dit_prepare.inputs = (text_to_dit_handle,)

        dit_to_aux_handle = ArtifactHandle(
            request_id=request_id,
            artifact_id="state_denoised_aux",
            kind=ArtifactKind.REQUEST_STATE,
            layout=ArtifactLayout.WORKER_LOCAL,
            producer_task_id=f"{request_id}:reshard_dit_to_aux:0",
            codec_id=state_codec_id,
        )
        dit_to_aux_task_id = f"{request_id}:reshard_dit_to_aux:0"
        tasks[dit_to_aux_task_id] = InferenceTask(
            task_id=dit_to_aux_task_id,
            request_id=request_id,
            kind=TaskKind.RESHARD,
            group_id=None,
            parallel_spec=ParallelSpec(),
            status=TaskStatus.PENDING,
            priority=priority,
            dependencies=(denoise_ids[-1],),
            inputs=(state_denoised,),
            outputs=(dit_to_aux_handle,),
            payload={
                RESHARD_PAYLOAD_SRC_GROUP_ID: last_dit_group_id,
                RESHARD_PAYLOAD_DST_GROUP_ID: aux_group_id,
                "request_stage": "reshard",
            },
        )
        vae_decode = tasks[f"{request_id}:vae_decode:0"]
        vae_decode.dependencies = (dit_to_aux_task_id,)
        vae_decode.inputs = (dit_to_aux_handle,)

    plan_metadata = {
        "request_type": request_type,
        "num_steps": num_steps,
        "chunk_size": chunk_size,
        "text_seq_len": text_seq_len,
        "latent_seq_len": latent_seq_len,
    }
    if metadata:
        plan_metadata.update(dict(metadata))
    if dit_step_schedule:
        plan_metadata["dit_step_schedule"] = tuple(dict(item) for item in dit_step_schedule)

    for task in tasks.values():
        payload = dict(task.payload)
        payload["request_metadata"] = dict(plan_metadata)
        task.payload = payload

    return RequestExecutionPlan(
        request_id=request_id,
        tasks=tasks,
        terminal_task_ids=(f"{request_id}:finalize:0",),
        initial_artifacts=(ArtifactValue(handle=req_handle, value=request_value),),
        metadata=plan_metadata,
    )
