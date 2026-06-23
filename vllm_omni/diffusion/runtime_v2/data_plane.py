"""RESHARD data plane: tensor slicing, migration schedule, and NCCL transfer.

This module owns all the logic that touches tensor layout, builds the migration
schedule from per-rank field descriptions, and performs the NCCL send/recv
choreography that moves artifacts between two execution groups. It also owns
the in-group SP gather/scatter wrappers used at task execution boundaries.

The worker (``_WorkerProcessRuntime``) keeps the multiprocessing pipe and the
task-execution loop; data-plane methods live here as a mixin so the worker
remains a thin orchestrator.
"""

from __future__ import annotations

import dataclasses
import threading
import time
import traceback
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import torch

from vllm_omni.diffusion.distributed.collective_runtime import (
    all_gather_tensor_list,
    get_gfc_migrate_runtime,
    runtime_v2_uses_gfc,
)
from vllm_omni.diffusion.distributed import parallel_state as omni_parallel_state
from vllm_omni.diffusion.runtime_v2._env import env_flag
from vllm_omni.diffusion.runtime_v2.codec import ArtifactLayoutCodec, ArtifactLayoutCodecRegistry
from vllm_omni.diffusion.runtime_v2.protocol import (
    ArtifactHandle,
    ArtifactMigrationSchedule,
    ArtifactValue,
    ExecutionGroupSpec,
    OutputArtifactLayout,
    TensorFieldLayout,
    TensorLayoutKind,
    TensorMigrationEdge,
    TensorSliceSpec,
)
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm_omni.diffusion.runtime_v2.multiproc_worker import (
        MigrateArtifactsCommand,
        MigrateArtifactsRankResult,
    )

logger = init_logger(__name__)

_MIGRATE_PHASE_DESCRIBE_LAYOUT = "describe_layout"
_MIGRATE_PHASE_BUILD_SCHEDULE = "build_schedule"
_MIGRATE_PHASE_ACCEPT_SCHEDULE = "accept_schedule"
_MIGRATE_PHASE_EXECUTE_TRANSFER = "execute_transfer"

# Env-gated sub-phase timing for execute_transfer. Off by default. When on, the
# transfer drains the device once at entry so the logged decomposition can
# separate pre-existing GPU backlog (drain_ms) from the NCCL transfer itself
# (nccl_gpu_ms). The entry drain serializes and perturbs timing, so this is a
# diagnostic flag only.
_MIGRATE_TRACE = env_flag("VLLM_RUNTIME_V2_MIGRATE_TRACE", False)

# Env-gated nsys capture for execute_transfer. Off by default. When on, NVTX
# ranges annotate the migration sub-phases and a cudaProfilerApi capture window
# opens once the first slow migration is seen, closing after a few are
# captured -- so an nsys run with --capture-range=cudaProfilerApi records only
# the slow-regime migrations rather than the whole run.
_MIGRATE_NSYS = env_flag("VLLM_RUNTIME_V2_MIGRATE_NSYS", False)
_nsys_capture = {"slow_seen": False, "active": False, "count": 0, "done": False}

# Route migration P2P through a dedicated GFC symmetric-memory runtime instead
# of NCCL isend/irecv. ON by default whenever the collective backend is GFC
# (see use_gfc below); set VLLM_RUNTIME_V2_MIGRATE_GFC_P2P=0 to force NCCL.
# GFC P2P avoids NCCL's lazy per-peer connection setup (cold ncclGroupEnd
# ~300-1200ms, which dominates migrations under SP-group churn -- on long
# elastic-SP runs the accumulated NCCL setup cost erases the policy's win). It
# runs on a SEPARATE GFC runtime (_GFC_MIGRATE_RUNTIME) so migration barriers
# never interleave with DiT collective barriers -- GFC requires a consistent
# per-rank-pair barrier order, which a runtime shared between the command
# thread (migration) and the exec thread (DiT) cannot guarantee.
_MIGRATE_GFC_P2P = env_flag("VLLM_RUNTIME_V2_MIGRATE_GFC_P2P", True)


def _gfc_p2p_chunked(
    gfc: Any, is_send: bool, peer_rank: int, buf: torch.Tensor
) -> None:
    """Move one migration edge buffer through GFC symmetric-memory P2P.

    GFC's comm buffers are allocated once at runtime init, so this pays no
    per-peer NCCL connection setup (the ncclGroupEnd / ncclCommInitRankConfig
    cost that dominates NCCL isend/irecv migrations under SP-group churn). A
    buffer larger than ``max_collective_bytes`` is split into chunks; src and
    dst derive identical chunk boundaries from the (matching) edge byte length,
    so their ``p2p_put`` / ``p2p_put_recv`` calls stay paired epoch-for-epoch.
    """
    flat = buf.reshape(-1).view(torch.uint8)
    total = int(flat.numel())
    max_bytes = int(gfc.config.max_collective_bytes)
    offset = 0
    while offset < total:
        n = min(max_bytes, total - offset)
        chunk = flat[offset : offset + n]
        if is_send:
            gfc.p2p_put(peer_rank, chunk)
        else:
            gfc.p2p_put_recv(peer_rank, chunk)
        offset += n


def parse_torch_dtype(dtype_str: str) -> torch.dtype:
    name = dtype_str
    if name.startswith("torch."):
        name = name[len("torch."):]
    candidate = getattr(torch, name, None)
    if not isinstance(candidate, torch.dtype):
        raise ValueError(f"unknown torch dtype: {dtype_str!r}")
    return candidate


# ---------------------------------------------------------------------------
# Pure tensor-slice helpers
# ---------------------------------------------------------------------------


def slice_tuple(offset: tuple[int, ...], shape: tuple[int, ...]) -> tuple[slice, ...]:
    return tuple(slice(int(start), int(start) + int(length)) for start, length in zip(offset, shape, strict=True))


def tensor_slice_view(tensor: torch.Tensor, spec: TensorSliceSpec) -> torch.Tensor:
    if tensor.dtype != parse_torch_dtype(spec.dtype):
        raise RuntimeError(
            "runtime_v2 tensor slice dtype mismatch: "
            f"field_path={spec.field_path}, tensor_dtype={tensor.dtype}, spec_dtype={spec.dtype}"
        )
    if len(spec.offset) != tensor.dim() or len(spec.shape) != tensor.dim():
        raise RuntimeError(
            "runtime_v2 tensor slice rank mismatch: "
            f"field_path={spec.field_path}, tensor_shape={tuple(tensor.shape)}, "
            f"offset={spec.offset}, shape={spec.shape}"
        )
    for offset, length, dim in zip(spec.offset, spec.shape, tensor.shape, strict=True):
        if int(offset) + int(length) > int(dim):
            raise RuntimeError(
                "runtime_v2 tensor slice exceeds tensor shape: "
                f"field_path={spec.field_path}, tensor_shape={tuple(tensor.shape)}, "
                f"offset={spec.offset}, shape={spec.shape}"
            )
    return tensor[slice_tuple(spec.offset, spec.shape)]


def copy_tensor_slice(
    *,
    dst: torch.Tensor,
    dst_offset: tuple[int, ...],
    src: torch.Tensor,
    src_offset: tuple[int, ...],
    shape: tuple[int, ...],
) -> None:
    if any(int(dim) == 0 for dim in shape):
        return
    dst_view = dst[slice_tuple(dst_offset, shape)]
    src_view = src[slice_tuple(src_offset, shape)]
    dst_view.copy_(src_view)


# ---------------------------------------------------------------------------
# Schedule-building math (no torch.distributed, no self state)
# ---------------------------------------------------------------------------


def tensor_numel(shape: tuple[int, ...]) -> int:
    numel = 1
    for dim in shape:
        numel *= int(dim)
    return numel


def validate_migration_field_pair(
    *,
    artifact_id: str,
    src_field: TensorFieldLayout,
    dst_field: TensorFieldLayout,
) -> None:
    supported = {TensorLayoutKind.REPLICATED, TensorLayoutKind.SHARDED}
    if src_field.layout.kind not in supported:
        raise NotImplementedError(
            "migration schedule does not support source tensor layout: "
            f"artifact_id={artifact_id}, field_path={src_field.field_path}, "
            f"layout={src_field.layout.kind.value}"
        )
    if dst_field.layout.kind not in supported:
        raise NotImplementedError(
            "migration schedule does not support destination tensor layout: "
            f"artifact_id={artifact_id}, field_path={dst_field.field_path}, "
            f"layout={dst_field.layout.kind.value}"
        )
    if src_field.dtype != dst_field.dtype:
        raise RuntimeError(
            "migration dtype mismatch: "
            f"artifact_id={artifact_id}, field_path={dst_field.field_path}, "
            f"src={src_field.dtype}, dst={dst_field.dtype}"
        )
    if tuple(src_field.global_shape) != tuple(dst_field.global_shape):
        raise RuntimeError(
            "migration global shape mismatch: "
            f"artifact_id={artifact_id}, field_path={dst_field.field_path}, "
            f"src={src_field.global_shape}, dst={dst_field.global_shape}"
        )


def intersect_field_slices(
    src_field: TensorFieldLayout,
    dst_field: TensorFieldLayout,
) -> tuple[tuple[int, ...], tuple[int, ...]] | None:
    start: list[int] = []
    shape: list[int] = []
    for src_offset, src_len, dst_offset, dst_len in zip(
        src_field.global_offset,
        src_field.local_shape,
        dst_field.global_offset,
        dst_field.local_shape,
        strict=True,
    ):
        inter_start = max(int(src_offset), int(dst_offset))
        inter_end = min(int(src_offset) + int(src_len), int(dst_offset) + int(dst_len))
        if inter_end <= inter_start:
            return None
        start.append(inter_start)
        shape.append(inter_end - inter_start)
    return tuple(start), tuple(shape)


def select_source_fields_for_dst(
    *,
    src_entries: list[tuple[int, TensorFieldLayout]],
    dst_rank: int,
    src_leader_rank: int,
) -> list[tuple[int, TensorFieldLayout]]:
    replicated = [
        (rank, field)
        for rank, field in src_entries
        if field.layout.kind == TensorLayoutKind.REPLICATED
    ]
    if replicated:
        for preferred_rank in (dst_rank, src_leader_rank):
            for rank, field in replicated:
                if rank == preferred_rank:
                    return [(rank, field)]
        return [min(replicated, key=lambda item: item[0])]

    deduped: dict[tuple[Any, ...], tuple[int, TensorFieldLayout]] = {}
    for rank, field in src_entries:
        key = (field.global_offset, field.local_shape, field.global_shape, field.dtype)
        previous = deduped.get(key)
        if previous is None:
            deduped[key] = (rank, field)
            continue
        previous_rank, _previous_field = previous
        if previous_rank == dst_rank:
            continue
        if rank == dst_rank or (previous_rank != src_leader_rank and rank == src_leader_rank) or rank < previous_rank:
            deduped[key] = (rank, field)
    return sorted(deduped.values(), key=lambda item: (tuple(item[1].global_offset), item[0]))


def select_replicated_source_field(
    *,
    field_path: tuple[str, ...],
    dst_rank: int,
    src_fields_by_rank: dict[int, dict[tuple[str, ...], TensorFieldLayout]],
    src_leader_rank: int,
) -> tuple[int, TensorFieldLayout]:
    if dst_rank in src_fields_by_rank and field_path in src_fields_by_rank[dst_rank]:
        return dst_rank, src_fields_by_rank[dst_rank][field_path]
    if src_leader_rank in src_fields_by_rank and field_path in src_fields_by_rank[src_leader_rank]:
        return src_leader_rank, src_fields_by_rank[src_leader_rank][field_path]
    for src_rank in sorted(src_fields_by_rank):
        src_field = src_fields_by_rank[src_rank].get(field_path)
        if src_field is not None:
            return src_rank, src_field
    raise RuntimeError(f"migration source field missing: field_path={field_path}")


def build_layout_migration_schedule(command: "MigrateArtifactsCommand") -> ArtifactMigrationSchedule:
    edges: list[TensorMigrationEdge] = []
    for artifact_spec in command.plan.artifacts:
        artifact_id = artifact_spec.handle.artifact_id
        src_layouts: dict[int, OutputArtifactLayout] = {}
        dst_layouts: dict[int, OutputArtifactLayout] = {}
        for result in command.layout_results:
            for layout in result.src_layouts:
                if layout.handle.artifact_id == artifact_id:
                    src_layouts[result.worker_rank] = layout
            for layout in result.dst_layouts:
                if layout.handle.artifact_id == artifact_id:
                    dst_layouts[result.worker_rank] = layout
        if not src_layouts:
            raise RuntimeError(f"migration source layouts missing for artifact {artifact_id}")
        if not dst_layouts:
            raise RuntimeError(f"migration destination layouts missing for artifact {artifact_id}")
        src_fields_by_path: dict[tuple[str, ...], list[tuple[int, TensorFieldLayout]]] = {}
        for src_rank, src_layout in src_layouts.items():
            for src_field in src_layout.tensors:
                src_fields_by_path.setdefault(src_field.field_path, []).append((src_rank, src_field))
        for dst_rank, dst_layout in dst_layouts.items():
            for dst_field in dst_layout.tensors:
                src_entries = src_fields_by_path.get(dst_field.field_path)
                if not src_entries:
                    raise RuntimeError(
                        "migration source field missing: "
                        f"artifact_id={artifact_id}, field_path={dst_field.field_path}"
                    )
                emitted = False
                for src_rank, src_field in select_source_fields_for_dst(
                    src_entries=src_entries,
                    dst_rank=dst_rank,
                    src_leader_rank=command.src_group.ranks[0],
                ):
                    validate_migration_field_pair(
                        artifact_id=artifact_id,
                        src_field=src_field,
                        dst_field=dst_field,
                    )
                    intersection = intersect_field_slices(src_field, dst_field)
                    if intersection is None:
                        continue
                    global_offset, shape = intersection
                    if any(int(dim) == 0 for dim in shape):
                        continue
                    src_offset = tuple(
                        int(start) - int(src_start)
                        for start, src_start in zip(global_offset, src_field.global_offset, strict=True)
                    )
                    dst_offset = tuple(
                        int(start) - int(dst_start)
                        for start, dst_start in zip(global_offset, dst_field.global_offset, strict=True)
                    )
                    edges.append(
                        TensorMigrationEdge(
                            artifact_id=artifact_id,
                            src_rank=src_rank,
                            dst_rank=dst_rank,
                            src_slice=TensorSliceSpec(
                                field_path=src_field.field_path,
                                offset=src_offset,
                                shape=shape,
                                dtype=src_field.dtype,
                            ),
                            dst_slice=TensorSliceSpec(
                                field_path=dst_field.field_path,
                                offset=dst_offset,
                                shape=shape,
                                dtype=dst_field.dtype,
                            ),
                            local_copy=src_rank == dst_rank,
                        )
                    )
                    emitted = True
                if not emitted and tensor_numel(dst_field.local_shape) > 0:
                    raise RuntimeError(
                        "migration source fields do not cover destination field: "
                        f"artifact_id={artifact_id}, field_path={dst_field.field_path}, "
                        f"dst_offset={dst_field.global_offset}, dst_shape={dst_field.local_shape}"
                    )
    participants = tuple(sorted(set(command.src_group.ranks) | set(command.dst_group.ranks)))
    return ArtifactMigrationSchedule(
        migrate_id=command.migrate_id,
        request_id=command.plan.request_id,
        src_group_id=command.src_group.group_id,
        dst_group_id=command.dst_group.group_id,
        src_ranks=tuple(command.src_group.ranks),
        dst_ranks=tuple(command.dst_group.ranks),
        participant_ranks=participants,
        edges=tuple(edges),
    )


# ---------------------------------------------------------------------------
# SP gather/scatter helpers (group-internal, used at task boundaries)
# ---------------------------------------------------------------------------


def all_gather_sharded_tensor(tensor: torch.Tensor, field: TensorFieldLayout) -> torch.Tensor:
    shard = field.layout.shard
    if shard is None:
        raise ValueError(f"missing shard spec for field_path={field.field_path}")
    sp_group = omni_parallel_state.get_sp_group()
    world_size = int(sp_group.world_size)
    if world_size == 1:
        return tensor
    shard_dim = int(shard.dim)
    if shard_dim < 0 or shard_dim >= tensor.dim():
        raise ValueError(f"invalid shard_dim={shard_dim} for tensor shape={tuple(tensor.shape)}")
    local_len = int(tensor.shape[shard_dim])
    size_tensor = torch.tensor([local_len], dtype=torch.int64, device=tensor.device)
    size_tensors = [torch.empty_like(size_tensor) for _ in range(world_size)]
    all_gather_tensor_list(size_tensors, size_tensor, group=sp_group.device_group)
    shard_lengths = [int(item.item()) for item in size_tensors]
    max_len = max(shard_lengths) if shard_lengths else 0
    padded_shape = list(tensor.shape)
    padded_shape[shard_dim] = max_len
    padded = torch.empty(tuple(padded_shape), dtype=tensor.dtype, device=tensor.device)
    if max_len > local_len:
        padded.zero_()
    if local_len > 0:
        copy_tensor_slice(
            dst=padded,
            dst_offset=tuple(0 for _ in padded_shape),
            src=tensor,
            src_offset=tuple(0 for _ in tensor.shape),
            shape=tuple(tensor.shape),
        )
    gathered = [torch.empty_like(padded) for _ in range(world_size)]
    all_gather_tensor_list(gathered, padded, group=sp_group.device_group)
    pieces = [
        chunk.narrow(shard_dim, 0, shard_len)
        for chunk, shard_len in zip(gathered, shard_lengths, strict=True)
        if shard_len > 0
    ]
    if pieces:
        full_tensor = torch.cat(pieces, dim=shard_dim).contiguous()
    else:
        full_tensor = torch.empty(tuple(field.global_shape), dtype=tensor.dtype, device=tensor.device)
    if tuple(full_tensor.shape) != tuple(field.global_shape):
        raise RuntimeError(
            "runtime_v2 sharded gather produced wrong shape: "
            f"field_path={field.field_path}, got={tuple(full_tensor.shape)}, expected={field.global_shape}"
        )
    return full_tensor


_SHARDED_FIELD_MISSING = 0
_SHARDED_FIELD_LOCAL = 1
_SHARDED_FIELD_GLOBAL = 2
_SHARDED_FIELD_INVALID = 3


def _status_collective_device(tensor: torch.Tensor | None) -> torch.device:
    if tensor is not None:
        return tensor.device
    if torch.cuda.is_available():
        return torch.device(f"cuda:{torch.cuda.current_device()}")
    return torch.device("cpu")


def _classify_sharded_field_tensor(tensor: Any, field: TensorFieldLayout) -> int:
    if tensor is None:
        return _SHARDED_FIELD_MISSING
    if not isinstance(tensor, torch.Tensor):
        return _SHARDED_FIELD_INVALID
    if tuple(tensor.shape) == tuple(field.local_shape):
        return _SHARDED_FIELD_LOCAL
    if tuple(tensor.shape) == tuple(field.global_shape):
        return _SHARDED_FIELD_GLOBAL
    return _SHARDED_FIELD_INVALID


def _gather_sharded_field_status(tensor: Any, field: TensorFieldLayout) -> list[int]:
    sp_group = omni_parallel_state.get_sp_group()
    world_size = int(sp_group.world_size)
    if world_size == 1:
        return [_classify_sharded_field_tensor(tensor, field)]
    device = _status_collective_device(tensor if isinstance(tensor, torch.Tensor) else None)
    status = torch.tensor(
        [_classify_sharded_field_tensor(tensor, field)],
        dtype=torch.int64,
        device=device,
    )
    statuses = [torch.empty_like(status) for _ in range(world_size)]
    all_gather_tensor_list(statuses, status, group=sp_group.device_group)
    return [int(item.item()) for item in statuses]


def _validate_sharded_field_status(
    *,
    artifact: ArtifactHandle,
    field: TensorFieldLayout,
    local_status: int,
    all_statuses: list[int],
    tensor: Any,
) -> bool:
    unique = set(all_statuses)
    if unique == {_SHARDED_FIELD_MISSING}:
        return False
    if _SHARDED_FIELD_INVALID in unique:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(
                "runtime_v2 sharded input field is not a tensor: "
                f"artifact_id={artifact.artifact_id}, field_path={field.field_path}, "
                f"got {type(tensor).__name__}"
            )
        raise RuntimeError(
            "runtime_v2 sharded input shape mismatch: "
            f"artifact_id={artifact.artifact_id}, field_path={field.field_path}, "
            f"tensor_shape={tuple(tensor.shape)}, local_shape={field.local_shape}, "
            f"global_shape={field.global_shape}, rank_statuses={all_statuses}"
        )
    if unique == {_SHARDED_FIELD_GLOBAL}:
        return False
    if unique != {_SHARDED_FIELD_LOCAL}:
        raise RuntimeError(
            "runtime_v2 sharded input has inconsistent per-rank materialization state: "
            f"artifact_id={artifact.artifact_id}, field_path={field.field_path}, "
            f"local_status={local_status}, rank_statuses={all_statuses}. "
            "All ranks in an SP group must either hold local shards or all hold full tensors."
        )
    return True


def materialize_input_artifact(
    *,
    artifact: ArtifactHandle,
    value: Any,
    group: ExecutionGroupSpec,
    group_rank: int,
    codec_registry: ArtifactLayoutCodecRegistry,
    request_metadata: Mapping[str, Any],
) -> None:
    """For sharded input fields, all-gather the local shard back to full tensor.

    The executor sees the full latent regardless of the storage SP layout; this
    is the dual of :func:`shard_output_artifact` below.
    """
    if not artifact.codec_id:
        return
    layout = codec_registry.describe_output(
        group=group,
        group_rank=group_rank,
        request_metadata=request_metadata,
        artifact=artifact,
    )
    for field in layout.tensors:
        if field.layout.kind != TensorLayoutKind.SHARDED:
            continue
        tensor = ArtifactLayoutCodec._read_field(value, field.field_path)
        statuses = _gather_sharded_field_status(tensor, field)
        local_status = statuses[group_rank]
        should_gather = _validate_sharded_field_status(
            artifact=artifact,
            field=field,
            local_status=local_status,
            all_statuses=statuses,
            tensor=tensor,
        )
        if not should_gather:
            continue
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(
                "runtime_v2 sharded input field is not a tensor: "
                f"artifact_id={artifact.artifact_id}, field_path={field.field_path}, "
                f"got {type(tensor).__name__}"
            )
        full_tensor = all_gather_sharded_tensor(tensor, field)
        ArtifactLayoutCodec._write_field(value, field.field_path, full_tensor)


def shard_output_artifact(
    *,
    artifact_value: ArtifactValue,
    group: ExecutionGroupSpec,
    group_rank: int,
    codec_registry: ArtifactLayoutCodecRegistry,
    request_metadata: Mapping[str, Any],
) -> ArtifactValue:
    """For sharded output fields, slice the executor's full tensor down to the rank's shard.

    Inverse of :func:`materialize_input_artifact`. Storage stays sharded so that
    RESHARD between groups can move only the slices that exist.
    """
    artifact = artifact_value.handle
    if not artifact.codec_id:
        return artifact_value
    layout = codec_registry.describe_output(
        group=group,
        group_rank=group_rank,
        request_metadata=request_metadata,
        artifact=artifact,
    )
    value = artifact_value.value
    for field in layout.tensors:
        if field.layout.kind != TensorLayoutKind.SHARDED:
            continue
        tensor = ArtifactLayoutCodec._read_field(value, field.field_path)
        if tensor is None:
            continue
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(
                "runtime_v2 sharded output field is not a tensor: "
                f"artifact_id={artifact.artifact_id}, field_path={field.field_path}, "
                f"got {type(tensor).__name__}"
            )
        if tuple(tensor.shape) == tuple(field.local_shape):
            continue
        if tuple(tensor.shape) != tuple(field.global_shape):
            raise RuntimeError(
                "runtime_v2 sharded output shape mismatch: "
                f"artifact_id={artifact.artifact_id}, field_path={field.field_path}, "
                f"tensor_shape={tuple(tensor.shape)}, local_shape={field.local_shape}, "
                f"global_shape={field.global_shape}"
            )
        local_tensor = tensor_slice_view(
            tensor,
            TensorSliceSpec(
                field_path=field.field_path,
                offset=field.global_offset,
                shape=field.local_shape,
                dtype=field.dtype,
            ),
        ).detach().contiguous()
        ArtifactLayoutCodec._write_field(value, field.field_path, local_tensor)
    return artifact_value


# ---------------------------------------------------------------------------
# Mixin: cross-group RESHARD phases (require worker-side state)
# ---------------------------------------------------------------------------


class MigrationDataPlaneMixin:
    """Cross-group RESHARD phases.

    Mix into a worker class that provides ``codec_registry`` /
    ``local_artifacts`` / ``_artifacts_lock`` / ``_migration_schedules`` /
    ``worker_rank`` / ``_artifact_key`` / ``_maybe_activate_group_session`` /
    ``_send_result`` / ``_group_spec``.
    """

    # ---- attributes the mixin reads on self -------------------------------
    if TYPE_CHECKING:
        codec_registry: ArtifactLayoutCodecRegistry
        local_artifacts: dict[tuple[str, str, str], ArtifactValue]
        _artifacts_lock: threading.RLock
        _migration_schedules: dict[str, ArtifactMigrationSchedule]
        worker_rank: int

        def _artifact_key(self, request_id: str, group_id: str, artifact_id: str) -> tuple[str, str, str]: ...
        def _maybe_activate_group_session(self, group_id: str) -> None: ...
        def _send_result(self, payload: Any) -> None: ...
        def _group_spec(self, group_id: str) -> ExecutionGroupSpec: ...

    # ---- per-task SP gather/scatter wrappers ------------------------------

    def _materialize_input_artifact(
        self,
        *,
        artifact: ArtifactHandle,
        value: Any,
        group_id: str,
        request_metadata: Mapping[str, Any],
    ) -> None:
        try:
            group = self._group_spec(group_id)
        except KeyError:
            return
        if self.worker_rank not in group.ranks:
            return
        materialize_input_artifact(
            artifact=artifact,
            value=value,
            group=group,
            group_rank=group.ranks.index(self.worker_rank),
            codec_registry=self.codec_registry,
            request_metadata=request_metadata,
        )

    def _shard_output_artifact(
        self,
        *,
        artifact_value: ArtifactValue,
        group_id: str,
        request_metadata: Mapping[str, Any],
    ) -> ArtifactValue:
        try:
            group = self._group_spec(group_id)
        except KeyError:
            return artifact_value
        if self.worker_rank not in group.ranks:
            return artifact_value
        return shard_output_artifact(
            artifact_value=artifact_value,
            group=group,
            group_rank=group.ranks.index(self.worker_rank),
            codec_registry=self.codec_registry,
            request_metadata=request_metadata,
        )

    # ---- RESHARD phase dispatch ------------------------------------------

    def _handle_migrate_artifacts(self, command: "MigrateArtifactsCommand") -> None:
        # Import here to avoid a top-level circular import with multiproc_worker.
        from vllm_omni.diffusion.runtime_v2.multiproc_worker import MigrateArtifactsRankResult

        started_ns = time.monotonic_ns()
        logger.info(
            "runtime_v2 worker op begin: rank=%s op=migrate_artifacts migrate_id=%s phase=%s mono_ns=%s",
            self.worker_rank,
            command.migrate_id,
            command.phase,
            started_ns,
        )
        try:
            if command.phase == _MIGRATE_PHASE_DESCRIBE_LAYOUT:
                result = self._describe_migration_layouts(command)
            elif command.phase == _MIGRATE_PHASE_BUILD_SCHEDULE:
                result = self._build_migration_schedule_result(command)
            elif command.phase == _MIGRATE_PHASE_ACCEPT_SCHEDULE:
                result = self._accept_migration_schedule(command)
            elif command.phase == _MIGRATE_PHASE_EXECUTE_TRANSFER:
                result = self._execute_migration_transfer(command)
            else:
                raise ValueError(f"unsupported migrate phase: {command.phase}")
        except Exception as exc:
            result = MigrateArtifactsRankResult(
                migrate_id=command.migrate_id,
                request_id=command.plan.request_id,
                worker_rank=self.worker_rank,
                phase=command.phase,
                error=f"{exc}\n{traceback.format_exc()}",
            )
        self._send_result(result)
        done_ns = time.monotonic_ns()
        logger.info(
            "runtime_v2 worker op done: rank=%s op=migrate_artifacts migrate_id=%s phase=%s status=%s "
            "elapsed_ns=%s mono_ns=%s",
            self.worker_rank,
            command.migrate_id,
            command.phase,
            "error" if result.error else "ok",
            max(0, done_ns - started_ns),
            done_ns,
        )

    def _describe_migration_layouts(
        self, command: "MigrateArtifactsCommand"
    ) -> "MigrateArtifactsRankResult":
        from vllm_omni.diffusion.runtime_v2.multiproc_worker import MigrateArtifactsRankResult

        src_layouts: list[OutputArtifactLayout] = []
        dst_layouts: list[OutputArtifactLayout] = []
        if self.worker_rank in command.src_group.ranks:
            self._maybe_activate_group_session(command.src_group.group_id)
            src_group_rank = command.src_group.ranks.index(self.worker_rank)
            for artifact_spec in command.plan.artifacts:
                src_handle = artifact_spec.effective_source_handle
                key = self._artifact_key(
                    command.plan.request_id,
                    command.src_group.group_id,
                    src_handle.artifact_id,
                )
                with self._artifacts_lock:
                    if key not in self.local_artifacts:
                        raise KeyError(
                            "source artifact not found for migration: "
                            f"request_id={command.plan.request_id}, group_id={command.src_group.group_id}, "
                            f"artifact_id={src_handle.artifact_id}, rank={self.worker_rank}"
                        )
                # Codec dispatch uses the src handle (codecs may select fields
                # by artifact_id substrings); the resulting layout is re-keyed
                # under the destination handle so build_schedule can match it
                # by destination artifact_id.
                actual_layout = self.codec_registry.describe_output(
                    group=command.src_group,
                    group_rank=src_group_rank,
                    request_metadata=command.plan.request_metadata,
                    artifact=src_handle,
                )
                src_layouts.append(
                    OutputArtifactLayout(handle=artifact_spec.handle, tensors=actual_layout.tensors)
                )
        if self.worker_rank in command.dst_group.ranks:
            self._maybe_activate_group_session(command.dst_group.group_id)
            dst_group_rank = command.dst_group.ranks.index(self.worker_rank)
            for artifact_spec in command.plan.artifacts:
                dst_layouts.append(
                    self.codec_registry.describe_output(
                        group=command.dst_group,
                        group_rank=dst_group_rank,
                        request_metadata=command.plan.request_metadata,
                        artifact=artifact_spec.handle,
                    )
                )
        return MigrateArtifactsRankResult(
            migrate_id=command.migrate_id,
            request_id=command.plan.request_id,
            worker_rank=self.worker_rank,
            phase=command.phase,
            src_layouts=tuple(src_layouts),
            dst_layouts=tuple(dst_layouts),
        )

    def _build_migration_schedule_result(
        self, command: "MigrateArtifactsCommand"
    ) -> "MigrateArtifactsRankResult":
        from vllm_omni.diffusion.runtime_v2.multiproc_worker import MigrateArtifactsRankResult

        if self.worker_rank != command.src_group.ranks[0]:
            raise RuntimeError(
                f"migration schedule must be built by src leader {command.src_group.ranks[0]}, "
                f"got rank {self.worker_rank}"
            )
        _trace = _MIGRATE_TRACE
        if _trace:
            _t0 = time.monotonic_ns()
        self._maybe_activate_group_session(command.src_group.group_id)
        if _trace:
            _t_act = time.monotonic_ns()
        schedule = build_layout_migration_schedule(command)
        if _trace:
            _t_build = time.monotonic_ns()
            _describe_ns = 0
            _pack_ns = 0
        _extra_edges: list[TensorMigrationEdge] = []
        # Pack non-tensor metadata for each artifact while we hold the source
        # value. The skeleton travels through the multiprocessing pipe (via the
        # controller broadcasting the EXECUTE_TRANSFER command) so dst ranks
        # can rebuild the full artifact value once tensors arrive over NCCL.
        src_group_rank = command.src_group.ranks.index(self.worker_rank)
        metadata_payloads: list[tuple[str, bytes]] = []
        for artifact_spec in command.plan.artifacts:
            dst_handle = artifact_spec.handle
            src_handle = artifact_spec.effective_source_handle
            key = self._artifact_key(
                command.plan.request_id,
                command.src_group.group_id,
                src_handle.artifact_id,
            )
            with self._artifacts_lock:
                value_holder = self.local_artifacts.get(key)
            if value_holder is None:
                raise KeyError(
                    "source artifact not found for migration metadata pack: "
                    f"request_id={command.plan.request_id}, group_id={command.src_group.group_id}, "
                    f"artifact_id={src_handle.artifact_id}, rank={self.worker_rank}"
                )
            codec = self.codec_registry.get(src_handle.codec_id or dst_handle.codec_id)
            if _trace:
                _d0 = time.monotonic_ns()
            layout = codec.describe_output(
                group=command.src_group,
                group_rank=src_group_rank,
                request_metadata=command.plan.request_metadata,
                artifact=src_handle,
            )
            if _trace:
                _d1 = time.monotonic_ns()
                _describe_ns += _d1 - _d0
            # On-device tensors the codec keeps outside describe_output (the
            # diffusers scheduler's solver history) migrate via P2P too: turn
            # each into a replicated edge (src leader -> every dst rank) so it
            # rides NCCL/GFC instead of a D2H + 40 MB pickle in pack_metadata.
            _src_leader = command.src_group.ranks[0]
            for _xf in codec.migration_extra_fields(value=value_holder.value, layout=layout):
                _xslice = TensorSliceSpec(
                    field_path=_xf.field_path,
                    offset=tuple(0 for _ in _xf.local_shape),
                    shape=_xf.local_shape,
                    dtype=_xf.dtype,
                )
                for _dst_rank in command.dst_group.ranks:
                    _extra_edges.append(
                        TensorMigrationEdge(
                            artifact_id=dst_handle.artifact_id,
                            src_rank=_src_leader,
                            dst_rank=int(_dst_rank),
                            src_slice=_xslice,
                            dst_slice=_xslice,
                            local_copy=(_src_leader == int(_dst_rank)),
                        )
                    )
            # Metadata is keyed by the destination artifact_id so dst ranks
            # can look it up under the same id they use everywhere else.
            _payload = codec.pack_metadata(value=value_holder.value, layout=layout)
            if _trace:
                _pack_ns += time.monotonic_ns() - _d1
            metadata_payloads.append((dst_handle.artifact_id, _payload))
        if _extra_edges:
            schedule = dataclasses.replace(
                schedule, edges=schedule.edges + tuple(_extra_edges)
            )
        self._migration_schedules[command.migrate_id] = schedule
        if _trace:
            logger.info(
                "runtime_v2 migrate build_schedule TRACE: rank=%s migrate_id=%s "
                "activate_ms=%.3f build_layout_ms=%.3f describe_ms=%.3f pack_ms=%.3f "
                "n_artifacts=%s n_edges=%s",
                self.worker_rank,
                command.migrate_id,
                (_t_act - _t0) / 1e6,
                (_t_build - _t_act) / 1e6,
                _describe_ns / 1e6,
                _pack_ns / 1e6,
                len(command.plan.artifacts),
                len(schedule.edges),
            )
        return MigrateArtifactsRankResult(
            migrate_id=command.migrate_id,
            request_id=command.plan.request_id,
            worker_rank=self.worker_rank,
            phase=command.phase,
            schedule=schedule,
            metadata_payloads=tuple(metadata_payloads),
        )

    def _accept_migration_schedule(
        self, command: "MigrateArtifactsCommand"
    ) -> "MigrateArtifactsRankResult":
        from vllm_omni.diffusion.runtime_v2.multiproc_worker import MigrateArtifactsRankResult

        if command.schedule is None:
            raise ValueError("accept_schedule phase requires a migration schedule")
        if self.worker_rank not in command.schedule.participant_ranks:
            raise RuntimeError(
                f"rank {self.worker_rank} is not a participant in migration {command.schedule.migrate_id}"
            )
        self._migration_schedules[command.migrate_id] = command.schedule
        return MigrateArtifactsRankResult(
            migrate_id=command.migrate_id,
            request_id=command.plan.request_id,
            worker_rank=self.worker_rank,
            phase=command.phase,
        )

    def _execute_migration_transfer(
        self, command: "MigrateArtifactsCommand"
    ) -> "MigrateArtifactsRankResult":
        # Real GPU group-to-group artifact migration. Each participating rank
        # posts NCCL P2P send/recv for the edges that touch it, then dst ranks
        # rebuild the full artifact value (skeleton + received tensors) and
        # publish it into their local_artifacts store under the dst group key.
        from vllm_omni.diffusion.runtime_v2.multiproc_worker import MigrateArtifactsRankResult

        schedule = command.schedule
        if schedule is None:
            raise ValueError("execute_transfer phase requires a schedule")
        if self.worker_rank not in schedule.participant_ranks:
            raise RuntimeError(
                f"rank {self.worker_rank} is not a participant in migration {schedule.migrate_id}"
            )
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            raise RuntimeError(
                "runtime_v2 RESHARD execute_transfer requires an initialized "
                "torch.distributed world group (NCCL); none is initialized"
            )
        if not torch.cuda.is_available():
            raise RuntimeError(
                "runtime_v2 RESHARD execute_transfer requires CUDA; no GPU is available "
                f"on rank {self.worker_rank}"
            )

        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        _trace = _MIGRATE_TRACE
        if _trace:
            _t0 = time.monotonic_ns()
            # Drain GPU work already queued before this migration (e.g. DiT
            # chunks the worker dispatched earlier) so the timings below
            # isolate the transfer itself from pre-existing backlog.
            torch.cuda.synchronize(device)
            _t_drain_ns = time.monotonic_ns()
            _drain_ns = _t_drain_ns - _t0
        _nsys = _MIGRATE_NSYS
        if _nsys:
            _nsys_t0 = time.monotonic_ns()
            if (
                _nsys_capture["slow_seen"]
                and not _nsys_capture["active"]
                and not _nsys_capture["done"]
            ):
                torch.cuda.profiler.start()
                _nsys_capture["active"] = True
                logger.info(
                    "runtime_v2 migrate NSYS capture START: rank=%s migrate_id=%s",
                    self.worker_rank,
                    command.migrate_id,
                )
            torch.cuda.nvtx.range_push(f"migrate_execute:{command.migrate_id[:8]}")
        is_src = self.worker_rank in command.src_group.ranks
        is_dst = self.worker_rank in command.dst_group.ranks
        if is_src:
            self._maybe_activate_group_session(command.src_group.group_id)
        elif is_dst:
            self._maybe_activate_group_session(command.dst_group.group_id)

        # Cache source values once per artifact to avoid repeated dict lookups.
        # Cache key is the destination artifact_id so it matches edge.artifact_id
        # downstream; the local_artifacts lookup uses the source artifact_id.
        src_values: dict[str, Any] = {}
        if is_src:
            for artifact_spec in command.plan.artifacts:
                src_handle = artifact_spec.effective_source_handle
                key = self._artifact_key(
                    command.plan.request_id,
                    command.src_group.group_id,
                    src_handle.artifact_id,
                )
                with self._artifacts_lock:
                    holder = self.local_artifacts.get(key)
                if holder is None:
                    raise KeyError(
                        "source artifact not found for migration execute: "
                        f"request_id={command.plan.request_id}, "
                        f"group_id={command.src_group.group_id}, "
                        f"artifact_id={src_handle.artifact_id}, "
                        f"rank={self.worker_rank}"
                    )
                src_values[artifact_spec.handle.artifact_id] = holder.value

        # Per-artifact dst-side assembled tensors keyed by field_path. For
        # sharded/irregular layouts each field can be filled by multiple edges.
        dst_field_tensors: dict[str, dict[tuple[str, ...], torch.Tensor]] = {}
        dst_layouts_by_artifact: dict[str, OutputArtifactLayout] = {}
        if is_dst:
            self._maybe_activate_group_session(command.dst_group.group_id)
            dst_group_rank = command.dst_group.ranks.index(self.worker_rank)
            for artifact_spec in command.plan.artifacts:
                codec = self.codec_registry.get(artifact_spec.handle.codec_id)
                dst_layout = codec.describe_output(
                    group=command.dst_group,
                    group_rank=dst_group_rank,
                    request_metadata=command.plan.request_metadata,
                    artifact=artifact_spec.handle,
                )
                dst_layouts_by_artifact[artifact_spec.handle.artifact_id] = dst_layout
                field_tensors: dict[tuple[str, ...], torch.Tensor] = {}
                for field in dst_layout.tensors:
                    field_tensors[field.field_path] = torch.empty(
                        tuple(field.local_shape),
                        dtype=parse_torch_dtype(field.dtype),
                        device=device,
                    )
                dst_field_tensors[artifact_spec.handle.artifact_id] = field_tensors
            # migration_extra_fields tensors (scheduler solver history) arrive
            # as schedule edges that describe_output did not declare. Allocate
            # their dst assemble-target slots from the edge specs so the recv
            # path and codec.assemble can find them by field_path.
            for _edge in schedule.edges:
                if _edge.dst_rank != self.worker_rank:
                    continue
                _slots = dst_field_tensors.setdefault(_edge.artifact_id, {})
                _fp = tuple(_edge.dst_slice.field_path)
                if _fp not in _slots:
                    _slots[_fp] = torch.empty(
                        tuple(_edge.dst_slice.shape),
                        dtype=parse_torch_dtype(_edge.dst_slice.dtype),
                        device=device,
                    )
        if _trace:
            _t_setup_ns = time.monotonic_ns()
        # Hold strong references to send buffers until the corresponding work
        # handle is waited on, so the tensor backing memory does not get freed
        # mid-transfer.
        send_tensors: list[torch.Tensor] = []
        recv_copies: list[tuple[torch.Tensor, TensorMigrationEdge]] = []
        # Sort edges deterministically so src and dst post matching pairs in
        # the same order (NCCL P2P matches multiple ops to the same peer FIFO).
        sorted_edges = sorted(
            schedule.edges,
            key=lambda e: (
                e.artifact_id,
                tuple(e.src_slice.field_path),
                e.src_rank,
                e.dst_rank,
                tuple(e.src_slice.offset),
                tuple(e.dst_slice.offset),
            ),
        )

        # NOTE: we intentionally do not use `torch.distributed.batch_isend_irecv`
        # here. On the NCCL backend that helper wraps the ops in
        # `_coalescing_manager`, whose end-of-window call requires *every* rank
        # of the process group (world) to have entered the same window, even
        # ranks that have no P2P op of their own. RESHARD's REPLICATED schedule
        # routes all sends out of a single src leader, so non-leader src-group
        # ranks end up with zero ops, skip the call, and deadlock the rest of
        # the participants in `_end_coalescing`. Posting each isend/irecv
        # independently sidesteps that requirement: each op is its own small
        # ncclGroupStart/End, matched pair-wise by peer, with no implicit
        # "all process-group ranks must enter" contract.
        if _trace:
            _evt_post_begin = torch.cuda.Event(enable_timing=True)
            _evt_post_end = torch.cuda.Event(enable_timing=True)
            _bytes_sent = 0
            _bytes_recv = 0
            _n_send = 0
            _n_recv = 0
            _n_local = 0
            _evt_post_begin.record()
            _t_post_begin_ns = time.monotonic_ns()
        if _nsys:
            torch.cuda.nvtx.range_push("migrate_post")
        # Optionally route migration P2P through the dedicated GFC migration
        # runtime instead of NCCL isend/irecv (opt-in via _MIGRATE_GFC_P2P).
        # GFC p2p avoids NCCL's lazy per-peer connection setup (cold
        # ncclGroupEnd ~300-1200ms); the dedicated runtime keeps migration
        # barriers isolated from DiT collectives.
        use_gfc = runtime_v2_uses_gfc() and _MIGRATE_GFC_P2P
        gfc_rt = get_gfc_migrate_runtime() if use_gfc else None
        p2p_ops: list[tuple[bool, int, torch.Tensor]] = []
        works: list[Any] = []
        for edge in sorted_edges:
            artifact_id = edge.artifact_id
            if any(int(dim) == 0 for dim in edge.src_slice.shape):
                continue
            if edge.local_copy:
                if not (edge.src_rank == self.worker_rank == edge.dst_rank):
                    continue
                tensor = ArtifactLayoutCodec._read_field(
                    src_values[artifact_id], tuple(edge.src_slice.field_path)
                )
                if not isinstance(tensor, torch.Tensor):
                    raise TypeError(
                        "RESHARD local-copy source field is not a tensor: "
                        f"artifact_id={artifact_id}, field_path={edge.src_slice.field_path}, "
                        f"got {type(tensor).__name__}"
                    )
                src_view = tensor_slice_view(tensor, edge.src_slice)
                dst_tensor = dst_field_tensors[artifact_id][tuple(edge.dst_slice.field_path)]
                copy_tensor_slice(
                    dst=dst_tensor,
                    dst_offset=edge.dst_slice.offset,
                    src=src_view,
                    src_offset=tuple(0 for _ in edge.src_slice.shape),
                    shape=edge.src_slice.shape,
                )
                if _trace:
                    _n_local += 1
                continue
            if edge.src_rank == self.worker_rank:
                tensor = ArtifactLayoutCodec._read_field(
                    src_values[artifact_id], tuple(edge.src_slice.field_path)
                )
                if not isinstance(tensor, torch.Tensor):
                    raise TypeError(
                        "RESHARD source field is not a tensor: "
                        f"artifact_id={artifact_id}, field_path={edge.src_slice.field_path}, "
                        f"got {type(tensor).__name__}"
                    )
                expected_dtype = parse_torch_dtype(edge.src_slice.dtype)
                if tensor.dtype != expected_dtype:
                    raise RuntimeError(
                        "RESHARD source dtype mismatch: "
                        f"artifact_id={artifact_id}, field_path={edge.src_slice.field_path}, "
                        f"src_value_dtype={tensor.dtype}, schedule_dtype={expected_dtype}"
                    )
                send_buffer = tensor_slice_view(tensor, edge.src_slice).detach().contiguous()
                if send_buffer.device != device:
                    # .to() to a different device already produces a fresh copy.
                    send_buffer = send_buffer.to(device, non_blocking=False)
                elif send_buffer.untyped_storage().data_ptr() == tensor.untyped_storage().data_ptr():
                    # .contiguous() is a no-op for an already-contiguous (e.g. full
                    # REPLICATED) slice, so send_buffer still aliases the live
                    # artifact's storage. The send is non-blocking (isend / GFC put)
                    # and only completes at work.wait()/stream.sync below; copy now so
                    # a concurrent in-place mutation of the artifact cannot corrupt the
                    # bytes on the wire.
                    send_buffer = send_buffer.clone()
                send_tensors.append(send_buffer)
                if use_gfc:
                    p2p_ops.append((True, edge.dst_rank, send_buffer))
                else:
                    works.append(torch.distributed.isend(send_buffer, edge.dst_rank))
                if _trace:
                    _n_send += 1
                    _bytes_sent += send_buffer.numel() * send_buffer.element_size()
            elif edge.dst_rank == self.worker_rank:
                dst_dtype = parse_torch_dtype(edge.dst_slice.dtype)
                recv_buffer = torch.empty(
                    tuple(edge.dst_slice.shape), dtype=dst_dtype, device=device
                )
                recv_copies.append((recv_buffer, edge))
                if use_gfc:
                    p2p_ops.append((False, edge.src_rank, recv_buffer))
                else:
                    works.append(torch.distributed.irecv(recv_buffer, edge.src_rank))
                if _trace:
                    _n_recv += 1
                    _bytes_recv += recv_buffer.numel() * recv_buffer.element_size()

        if use_gfc and p2p_ops:
            # Pre-register the 2-rank pair groups (idempotent; writes the rank
            # row on the current stream) before handing off to GFC's stream,
            # so GFC's barrier kernels observe both the row and the send
            # buffers built in the loop above.
            for _is_send, _peer, _buf in p2p_ops:
                pair = (
                    (self.worker_rank, _peer)
                    if _is_send
                    else (_peer, self.worker_rank)
                )
                gfc_rt.register_group(pair)
            gfc_rt.stream.wait_stream(torch.cuda.current_stream(device))
            for _is_send, _peer, _buf in p2p_ops:
                _gfc_p2p_chunked(gfc_rt, _is_send, _peer, _buf)
        if _trace:
            _evt_post_end.record()
            _t_post_end_ns = time.monotonic_ns()
        if _nsys:
            torch.cuda.nvtx.range_pop()  # migrate_post
        for work in works:
            work.wait()
        if _trace:
            _t_wait_end_ns = time.monotonic_ns()
        if _nsys:
            torch.cuda.nvtx.range_push("migrate_sync")
        if use_gfc:
            if p2p_ops:
                gfc_rt.stream.synchronize()
        elif works:
            torch.cuda.synchronize(device)
        if _nsys:
            torch.cuda.nvtx.range_pop()  # migrate_sync
        if _trace:
            _t_sync_end_ns = time.monotonic_ns()

        for recv_buffer, edge in recv_copies:
            dst_tensor = dst_field_tensors[edge.artifact_id][tuple(edge.dst_slice.field_path)]
            copy_tensor_slice(
                dst=dst_tensor,
                dst_offset=edge.dst_slice.offset,
                src=recv_buffer,
                src_offset=tuple(0 for _ in edge.dst_slice.shape),
                shape=edge.dst_slice.shape,
            )

        # Assemble values on dst ranks. For replicated layouts each rank holds
        # a full tensor; for sharded layouts each rank holds its local slice.
        if is_dst:
            metadata_lookup = dict(command.metadata_payloads)
            for artifact_spec in command.plan.artifacts:
                artifact_id = artifact_spec.handle.artifact_id
                metadata_bytes = metadata_lookup.get(artifact_id)
                if metadata_bytes is None:
                    raise RuntimeError(
                        "missing metadata payload for migration execute: "
                        f"migrate_id={command.migrate_id}, artifact_id={artifact_id}, "
                        f"rank={self.worker_rank}"
                    )
                tensors = dst_field_tensors.get(artifact_id, {})
                codec = self.codec_registry.get(artifact_spec.handle.codec_id)
                dst_layout = dst_layouts_by_artifact[artifact_id]
                value = codec.assemble(
                    metadata_bytes=metadata_bytes,
                    tensors=tensors,
                    layout=dst_layout,
                    device=device,
                )
                key = self._artifact_key(
                    command.plan.request_id,
                    command.dst_group.group_id,
                    artifact_id,
                )
                with self._artifacts_lock:
                    self.local_artifacts[key] = ArtifactValue(
                        handle=artifact_spec.handle, value=value
                    )

        # Free the schedule once execute completes; controller will not retry.
        self._migration_schedules.pop(command.migrate_id, None)

        if _trace:
            _t_assemble_end_ns = time.monotonic_ns()
            # Both events completed once the synchronize above drained the
            # device; since the device was also drained at entry, this is the
            # pure GPU duration of the transfer region with no backlog ahead.
            _nccl_gpu_ms = (
                _evt_post_begin.elapsed_time(_evt_post_end) if works else 0.0
            )
            logger.info(
                "runtime_v2 migrate execute_transfer TRACE: rank=%s migrate_id=%s "
                "is_src=%s is_dst=%s total_ms=%.3f drain_ms=%.3f setup_ms=%.3f "
                "post_cpu_ms=%.3f wait_ms=%.3f sync_ms=%.3f assemble_ms=%.3f "
                "nccl_gpu_ms=%.3f n_edges=%s n_send=%s n_recv=%s n_local=%s "
                "bytes_sent=%s bytes_recv=%s mono_ns=%s",
                self.worker_rank,
                command.migrate_id,
                is_src,
                is_dst,
                (_t_assemble_end_ns - _t0) / 1e6,
                _drain_ns / 1e6,
                (_t_setup_ns - _t_drain_ns) / 1e6,
                (_t_post_end_ns - _t_post_begin_ns) / 1e6,
                (_t_wait_end_ns - _t_post_end_ns) / 1e6,
                (_t_sync_end_ns - _t_wait_end_ns) / 1e6,
                (_t_assemble_end_ns - _t_sync_end_ns) / 1e6,
                _nccl_gpu_ms,
                len(sorted_edges),
                _n_send,
                _n_recv,
                _n_local,
                _bytes_sent,
                _bytes_recv,
                _t_assemble_end_ns,
            )

        if _nsys:
            torch.cuda.nvtx.range_pop()  # migrate_execute
            _nsys_total_ms = (time.monotonic_ns() - _nsys_t0) / 1e6
            if _nsys_total_ms > 200.0:
                _nsys_capture["slow_seen"] = True
            if _nsys_capture["active"]:
                _nsys_capture["count"] += 1
                if _nsys_capture["count"] >= 15:
                    torch.cuda.profiler.stop()
                    _nsys_capture["active"] = False
                    _nsys_capture["done"] = True
                    logger.info(
                        "runtime_v2 migrate NSYS capture STOP: rank=%s captured=%s",
                        self.worker_rank,
                        _nsys_capture["count"],
                    )

        return MigrateArtifactsRankResult(
            migrate_id=command.migrate_id,
            request_id=command.plan.request_id,
            worker_rank=self.worker_rank,
            phase=command.phase,
        )
