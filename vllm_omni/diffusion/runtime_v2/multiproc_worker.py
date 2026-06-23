from __future__ import annotations

import contextlib
import copy
import multiprocessing as mp
import os
import pickle
import queue
import threading
import time
import traceback
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from multiprocessing.connection import Connection, wait
from typing import Any
import uuid

import torch
import vllm.distributed.parallel_state as vllm_parallel_state
from vllm.logger import init_logger

from vllm_omni.diffusion.data import DiffusionOutput, DiffusionParallelConfig, OmniDiffusionConfig
from vllm_omni.diffusion.distributed import parallel_state as omni_parallel_state
from vllm_omni.diffusion.forward_context import set_forward_context
from vllm_omni.diffusion.ipc import pack_diffusion_output_shm, unpack_diffusion_output_shm
from vllm_omni.diffusion.profiler import CurrentProfiler
from vllm_omni.diffusion.runtime_v2._env import env_flag
from vllm_omni.diffusion.runtime_v2.codec import ArtifactLayoutCodecRegistry
from vllm_omni.diffusion.runtime_v2.data_plane import (
    MigrationDataPlaneMixin,
)
from vllm_omni.diffusion.runtime_v2.interfaces import WorkerExecutor
from vllm_omni.diffusion.runtime_v2.migration_engine import MigrationEngine, MigrationJob
from vllm_omni.diffusion.runtime_v2.protocol import (
    ArtifactMigrationSchedule,
    ArtifactKind,
    ArtifactValue,
    ExecutionGroupSpec,
    InferenceTask,
    MigrateArtifactsPlan,
    OutputArtifactLayout,
    ParallelSpec,
    TaskKind,
    WorkerEvent,
    WorkerEventKind,
    WorkerLocalArtifactRef,
)
from vllm_omni.diffusion.runtime_v2.registry import get_runtime_v2_adapter
from vllm_omni.diffusion.runtime_v2.topology import RuntimeTopology
from vllm_omni.diffusion.worker.diffusion_worker import WorkerWrapperBase

logger = init_logger(__name__)
_CPU_THREAD_TRACE_MIN_NS = 10_000_000


@dataclass(frozen=True)
class SerializedArtifactValue:
    handle: Any
    payload: bytes
    transport: str = "pickle"
    payload_nbytes: int = 0

    @property
    def value(self) -> Any:
        return _decode_serialized_payload(self)


@dataclass(frozen=True)
class ProcessDispatchTaskCommand:
    task: InferenceTask
    inline_inputs: tuple[SerializedArtifactValue, ...]
    result_owner_rank: int
    release_after_exec_artifact_ids: tuple[str, ...] = ()
    dispatched_at_ns: int = 0
    group_spec: ExecutionGroupSpec | None = None


@dataclass(frozen=True)
class ShutdownWorkerCommand:
    reason: str = ""


@dataclass(frozen=True)
class FetchArtifactsCommand:
    request_id: str
    group_id: str
    artifact_ids: tuple[str, ...]
    fetch_id: str = ""


@dataclass(frozen=True)
class MigrateArtifactsCommand:
    migrate_id: str
    phase: str
    plan: MigrateArtifactsPlan
    src_group: ExecutionGroupSpec
    dst_group: ExecutionGroupSpec
    layout_results: tuple["MigrateArtifactsRankResult", ...] = ()
    schedule: ArtifactMigrationSchedule | None = None
    # Per-artifact pickled skeleton produced by the src leader during BUILD_SCHEDULE.
    # Populated for the EXECUTE_TRANSFER phase; ignored otherwise.
    metadata_payloads: tuple[tuple[str, bytes], ...] = ()


@dataclass(frozen=True)
class EvictRequestCommand:
    request_id: str


@dataclass(frozen=True)
class FetchArtifactsResult:
    request_id: str
    worker_rank: int
    artifacts: tuple[SerializedArtifactValue | ArtifactValue, ...] = ()
    error: str | None = None
    fetch_id: str = ""


@dataclass(frozen=True)
class MigrateArtifactsRankResult:
    migrate_id: str
    request_id: str
    worker_rank: int
    phase: str
    src_layouts: tuple[OutputArtifactLayout, ...] = ()
    dst_layouts: tuple[OutputArtifactLayout, ...] = ()
    schedule: ArtifactMigrationSchedule | None = None
    # Per-artifact pickled skeleton; only set by src leader during BUILD_SCHEDULE.
    metadata_payloads: tuple[tuple[str, bytes], ...] = ()
    error: str | None = None


@dataclass(frozen=True)
class RpcCommand:
    rpc_id: str
    method: str
    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] | None = None
    reply: bool = True


@dataclass(frozen=True)
class RpcResult:
    rpc_id: str
    worker_rank: int
    result: Any | None = None
    error: str | None = None


@dataclass(frozen=True)
class WorkerReadyMessage:
    worker_rank: int
    status: str = "ready"
    message: str = ""


@dataclass(frozen=True)
class _FixedParallelSession:
    parallel_spec: ParallelSpec
    world: Any
    dp: Any
    cfg: Any
    pp: Any
    sp: Any
    tp: Any
    fs: Any
    dit: Any
    ep: Any

    @classmethod
    def capture_current(cls, parallel_spec: ParallelSpec) -> "_FixedParallelSession":
        session = cls(
            parallel_spec=parallel_spec,
            world=omni_parallel_state._WORLD,
            dp=omni_parallel_state._DP,
            cfg=omni_parallel_state._CFG,
            pp=omni_parallel_state._PP,
            sp=omni_parallel_state._SP,
            tp=vllm_parallel_state._TP,
            fs=omni_parallel_state._FS,
            dit=omni_parallel_state._DIT,
            ep=getattr(vllm_parallel_state, "_EP", None),
        )
        session.validate()
        return session

    def validate(self) -> None:
        if self.world is None:
            raise RuntimeError("runtime_v2 worker session requires initialized distributed world group")
        if self.dp is None or self.cfg is None or self.pp is None or self.sp is None or self.tp is None:
            raise RuntimeError("runtime_v2 worker session requires initialized legacy parallel groups")
        if self.tp.world_size != int(self.parallel_spec.tp):
            raise RuntimeError(
                f"legacy TP world_size={self.tp.world_size} does not match runtime_v2 tp={self.parallel_spec.tp}"
            )
        if self.sp.world_size != int(self.parallel_spec.sp):
            raise RuntimeError(
                f"legacy SP world_size={self.sp.world_size} does not match runtime_v2 sp={self.parallel_spec.sp}"
            )
        if self.cfg.world_size != int(self.parallel_spec.cfg):
            raise RuntimeError(
                f"legacy CFG world_size={self.cfg.world_size} does not match runtime_v2 cfg={self.parallel_spec.cfg}"
            )

    def activate(self) -> None:
        omni_parallel_state._WORLD = self.world
        omni_parallel_state._DP = self.dp
        omni_parallel_state._CFG = self.cfg
        omni_parallel_state._PP = self.pp
        omni_parallel_state._SP = self.sp
        omni_parallel_state._FS = self.fs
        omni_parallel_state._DIT = self.dit
        vllm_parallel_state._DP = self.dp
        vllm_parallel_state._PP = self.pp
        vllm_parallel_state._TP = self.tp
        if hasattr(vllm_parallel_state, "_EP"):
            vllm_parallel_state._EP = self.ep


def _clone_diffusion_output_for_transport(output: DiffusionOutput) -> DiffusionOutput:
    return DiffusionOutput(
        output=output.output,
        trajectory_timesteps=output.trajectory_timesteps,
        trajectory_latents=output.trajectory_latents,
        trajectory_decoded=output.trajectory_decoded,
        error=output.error,
        post_process_func=output.post_process_func,
        custom_output=output.custom_output,
    )


def _diffusion_output_uses_shm_handles(output: DiffusionOutput) -> bool:
    return (
        isinstance(output.output, dict)
        and bool(output.output.get("__tensor_shm__"))
        or isinstance(output.trajectory_latents, dict)
        and bool(output.trajectory_latents.get("__tensor_shm__"))
    )


def _decode_serialized_payload(artifact_value: SerializedArtifactValue, *, unpack_shm: bool = True) -> Any:
    value = pickle.loads(artifact_value.payload)
    if unpack_shm and artifact_value.transport == "shm" and isinstance(value, DiffusionOutput):
        unpack_diffusion_output_shm(value)
    return value


def _serialize_artifact_value(
    artifact_value: ArtifactValue,
    *,
    prefer_shm_output: bool = False,
) -> SerializedArtifactValue:
    if (
        prefer_shm_output
        and artifact_value.handle.kind == ArtifactKind.OUTPUT
        and isinstance(artifact_value.value, DiffusionOutput)
    ):
        try:
            output_for_transport = _clone_diffusion_output_for_transport(artifact_value.value)
            pack_diffusion_output_shm(output_for_transport)
            if _diffusion_output_uses_shm_handles(output_for_transport):
                payload = pickle.dumps(output_for_transport, protocol=pickle.HIGHEST_PROTOCOL)
                return SerializedArtifactValue(
                    handle=artifact_value.handle,
                    payload=payload,
                    transport="shm",
                    payload_nbytes=len(payload),
                )
        except Exception as exc:
            try:
                unpack_diffusion_output_shm(output_for_transport)
            except Exception:
                pass
            logger.warning(
                "runtime_v2 fetch_artifacts shm serialize fallback to pickle: request_id=%s artifact_id=%s error=%s",
                artifact_value.handle.request_id,
                artifact_value.handle.artifact_id,
                exc,
            )

    payload = pickle.dumps(artifact_value.value, protocol=pickle.HIGHEST_PROTOCOL)
    return SerializedArtifactValue(
        handle=artifact_value.handle,
        payload=payload,
        transport="pickle",
        payload_nbytes=len(payload),
    )


def _deserialize_artifact_value(
    artifact_value: SerializedArtifactValue,
    *,
    unpack_shm: bool = True,
) -> ArtifactValue:
    return ArtifactValue(
        handle=artifact_value.handle,
        value=_decode_serialized_payload(artifact_value, unpack_shm=unpack_shm),
    )


def _clone_runtime_worker_config(od_config: OmniDiffusionConfig) -> OmniDiffusionConfig:
    worker_config = copy.deepcopy(od_config)
    worker_config.enable_runtime_v2 = False
    return worker_config


class _WorkerProcessRuntime(MigrationDataPlaneMixin):
    def __init__(
        self,
        *,
        worker_rank: int,
        device_id: int | None,
        od_config: OmniDiffusionConfig,
        parallel_spec: ParallelSpec,
        dist_rank: int | None = None,
        local_rank: int | None = None,
        world_size: int | None = None,
        master_port: int | None = None,
        group_id: str = "g0",
        command_pipe_r: Connection,
        event_pipe_w: Connection,
        result_pipe_w: Connection,
    ) -> None:
        self.worker_rank = worker_rank
        self.device_id = device_id
        self.od_config = od_config
        self.parallel_spec = parallel_spec
        self.dist_rank = int(worker_rank if dist_rank is None else dist_rank)
        self.local_rank = int(self.dist_rank if local_rank is None else local_rank)
        configured_world_size = getattr(od_config, "num_gpus", None) if world_size is None else world_size
        configured_master_port = getattr(od_config, "master_port", None) if master_port is None else master_port
        self.world_size = self._safe_positive_int(configured_world_size, default=1)
        self.master_port = self._safe_positive_int(configured_master_port, default=30005)
        self.group_id = group_id
        self.command_pipe_r = command_pipe_r
        self.event_pipe_w = event_pipe_w
        self.result_pipe_w = result_pipe_w
        self.worker_wrapper: WorkerWrapperBase | None = None
        self.executors: dict[Any, WorkerExecutor] = {}
        self.codec_registry = ArtifactLayoutCodecRegistry()
        self.local_artifacts: dict[tuple[str, str, str], ArtifactValue] = {}
        self._migration_schedules: dict[str, ArtifactMigrationSchedule] = {}
        self.fixed_session: _FixedParallelSession | None = None
        self.fixed_sessions: dict[str, _FixedParallelSession] = {}
        self._execution_group_specs_by_id: dict[str, ExecutionGroupSpec] = {}
        self._parallel_config_by_group: dict[str, DiffusionParallelConfig] = {}
        # group_id whose parallel_config is currently applied across the model
        # tree. Lets _apply_group_parallel_config skip the full DiT submodule
        # walk when the active group is unchanged (the common case: every dit
        # step of a request between migrations, and all of a static lane).
        self._active_parallel_group_id: str | None = None
        # Cached list of parallel-aware modules (transformer + submodules that
        # carry a parallel_config). module.modules() over the 20B DiT is the
        # expensive part of an apply; the tree is fixed after load so we
        # enumerate once and reuse. Reset when the pipeline is rebuilt.
        self._pcfg_target_modules: list | None = None
        self._artifacts_lock = threading.RLock()
        self._result_pipe_lock = threading.Lock()
        self._fetch_queue: queue.Queue[FetchArtifactsCommand | None] = queue.Queue()
        self._fetch_thread: threading.Thread | None = None
        # DiT chunks awaiting GPU-timing readout: tuples of (task_id, group_id,
        # cuda_start_event, cuda_end_event, activate_ms, cpu_ms). Filled in
        # _execute_task, drained by _drain_dit_timing on the run loop once the
        # CUDA end event has completed.
        self._pending_dit_timing: list[tuple[Any, ...]] = []
        self._fetch_stop = threading.Event()
        self._fetch_copy_stream: Any | None = None
        self._auto_profile_first_n_reqs = self._read_non_negative_int_env("VLLM_DIFFUSION_PROFILE_FIRST_N_REQS", 0)
        self._auto_profile_trace_prefix = os.environ.get("VLLM_DIFFUSION_PROFILE_TRACE_PREFIX", "runtime_v2_auto")
        self._auto_profile_started = False
        self._auto_profile_stopped = False
        self._auto_profile_template: str | None = None
        self._auto_profile_completed_requests: set[str] = set()
        self._fetch_window_profile_enabled = env_flag("VLLM_RUNTIME_V2_PROFILE_FETCH_WINDOW", False)
        self._fetch_window_profile_target_request_id = (
            os.environ.get("VLLM_RUNTIME_V2_PROFILE_FETCH_WINDOW_REQUEST_ID", "").strip() or None
        )
        self._fetch_window_profile_dispatch_target = self._read_non_negative_int_env(
            "VLLM_RUNTIME_V2_PROFILE_FETCH_WINDOW_DISPATCHES",
            2,
        )
        self._fetch_window_profile_active = False
        self._fetch_window_profile_start_ns = 0
        self._fetch_window_profile_request_id: str | None = None
        self._fetch_window_profile_remaining_dispatches = 0
        self._fetch_window_profile_seq = 0

    @staticmethod
    def _safe_positive_int(value: Any, *, default: int) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return default
        if parsed <= 0:
            return default
        return parsed

    @staticmethod
    def _command_name(command: Any) -> str:
        if isinstance(command, ShutdownWorkerCommand):
            return "shutdown"
        if isinstance(command, ProcessDispatchTaskCommand):
            return "dispatch_task"
        if isinstance(command, FetchArtifactsCommand):
            return "fetch_artifacts"
        if isinstance(command, MigrateArtifactsCommand):
            return f"migrate_artifacts:{command.phase}"
        if isinstance(command, EvictRequestCommand):
            return "evict_request"
        if isinstance(command, RpcCommand):
            return "rpc"
        return type(command).__name__

    @staticmethod
    def _read_non_negative_int_env(name: str, default: int) -> int:
        raw = os.environ.get(name, "").strip()
        if not raw:
            return default
        try:
            value = int(raw)
        except ValueError:
            logger.warning("runtime_v2 worker invalid env %s=%r, fallback=%s", name, raw, default)
            return default
        if value < 0:
            logger.warning("runtime_v2 worker invalid env %s=%s (<0), fallback=%s", name, value, default)
            return default
        return value

    @staticmethod
    def _artifact_key(request_id: str, group_id: str, artifact_id: str) -> tuple[str, str, str]:
        return (request_id, group_id, artifact_id)

    @staticmethod
    def _require_task_group_id(task: InferenceTask) -> str:
        if task.group_id is None:
            raise ValueError(f"task {task.task_id} has no group_id")
        return task.group_id

    def _build_execution_group_specs_by_id(self) -> dict[str, ExecutionGroupSpec]:
        specs: dict[str, ExecutionGroupSpec] = {}
        raw_groups = getattr(self.od_config, "runtime_v2_execution_groups", None) or ()
        for raw in raw_groups:
            group_id = str(raw.get("group_id", ""))
            if not group_id:
                continue
            tp = int(raw.get("tp", 1))
            sp = int(raw.get("sp", 1))
            cfg = int(raw.get("cfg", 1))
            specs[group_id] = ExecutionGroupSpec(
                group_id=group_id,
                ranks=tuple(int(rank) for rank in raw.get("ranks", ())),
                parallel_spec=ParallelSpec(tp=tp, sp=sp, cfg=cfg),
                supported_task_kinds=tuple(TaskKind),
                ulysses_degree=int(raw.get("ulysses_degree", sp)),
                ring_degree=int(raw.get("ring_degree", 1)),
            )
        if self.group_id not in specs:
            base_parallel = getattr(self.od_config, "parallel_config", None)
            specs[self.group_id] = ExecutionGroupSpec(
                group_id=self.group_id,
                ranks=tuple(range(int(self.world_size))),
                parallel_spec=self.parallel_spec,
                supported_task_kinds=tuple(TaskKind),
                ulysses_degree=int(getattr(base_parallel, "ulysses_degree", 1)),
                ring_degree=int(getattr(base_parallel, "ring_degree", 1)),
            )
        return specs

    def _build_parallel_configs_by_group(self) -> dict[str, DiffusionParallelConfig]:
        base_parallel = getattr(self.od_config, "parallel_config", None)
        configs: dict[str, DiffusionParallelConfig] = {}
        for group_id, spec in self._execution_group_specs_by_id.items():
            configs[group_id] = DiffusionParallelConfig(
                pipeline_parallel_size=1,
                data_parallel_size=1,
                tensor_parallel_size=int(spec.parallel_spec.tp),
                enable_expert_parallel=bool(getattr(base_parallel, "enable_expert_parallel", False)),
                sequence_parallel_size=int(spec.parallel_spec.sp),
                ulysses_degree=int(spec.ulysses_degree),
                ring_degree=int(spec.ring_degree),
                cfg_parallel_size=int(spec.parallel_spec.cfg),
                vae_patch_parallel_size=int(getattr(base_parallel, "vae_patch_parallel_size", 1)),
                use_hsdp=False,
                hsdp_shard_size=-1,
                hsdp_replicate_size=1,
            )
        return configs

    def _group_spec(self, group_id: str) -> ExecutionGroupSpec:
        try:
            return self._execution_group_specs_by_id[str(group_id)]
        except KeyError:
            raise KeyError(f"runtime_v2 worker has no execution group spec for group_id={group_id!r}") from None

    @staticmethod
    def _group_spec_payload(spec: ExecutionGroupSpec) -> dict[str, Any]:
        return {
            "group_id": str(spec.group_id),
            "ranks": [int(rank) for rank in spec.ranks],
            "tp": int(spec.parallel_spec.tp),
            "sp": int(spec.parallel_spec.sp),
            "cfg": int(spec.parallel_spec.cfg),
            "ulysses_degree": int(spec.ulysses_degree),
            "ring_degree": int(spec.ring_degree),
        }

    @staticmethod
    def _topology_signature(spec: ExecutionGroupSpec) -> tuple[Any, ...]:
        # Worker-side parallel-session construction only cares about topology
        # (ranks + parallel dims). supported_task_kinds is scheduler routing
        # metadata and may legitimately differ between the worker's view (built
        # from runtime_v2_execution_groups with all kinds) and the scheduler's
        # view (built from the adapter's STANDARD_TASK_KINDS).
        return (
            tuple(int(rank) for rank in spec.ranks),
            int(spec.parallel_spec.tp),
            int(spec.parallel_spec.sp),
            int(spec.parallel_spec.cfg),
            int(spec.ulysses_degree),
            int(spec.ring_degree),
        )

    def _ensure_group_spec(self, spec: ExecutionGroupSpec) -> None:
        existing = self._execution_group_specs_by_id.get(spec.group_id)
        if existing is not None:
            if self._topology_signature(existing) != self._topology_signature(spec):
                raise ValueError(
                    f"runtime_v2 worker group {spec.group_id!r} already exists with a different topology: "
                    f"existing={self._topology_signature(existing)!r} incoming={self._topology_signature(spec)!r}"
                )
            return
        base_parallel = getattr(self.od_config, "parallel_config", None)
        self._execution_group_specs_by_id[spec.group_id] = spec
        self._parallel_config_by_group[spec.group_id] = DiffusionParallelConfig(
            pipeline_parallel_size=1,
            data_parallel_size=1,
            tensor_parallel_size=int(spec.parallel_spec.tp),
            enable_expert_parallel=bool(getattr(base_parallel, "enable_expert_parallel", False)),
            sequence_parallel_size=int(spec.parallel_spec.sp),
            ulysses_degree=int(spec.ulysses_degree),
            ring_degree=int(spec.ring_degree),
            cfg_parallel_size=int(spec.parallel_spec.cfg),
            vae_patch_parallel_size=int(getattr(base_parallel, "vae_patch_parallel_size", 1)),
            use_hsdp=False,
            hsdp_shard_size=-1,
            hsdp_replicate_size=1,
        )
        if self.worker_rank not in spec.ranks:
            return
        omni_parallel_state.ensure_runtime_v2_model_parallel_session(
            self._group_spec_payload(spec),
        )
        omni_parallel_state.activate_runtime_v2_model_parallel_session(spec.group_id)
        self.fixed_sessions[spec.group_id] = _FixedParallelSession.capture_current(spec.parallel_spec)

    @staticmethod
    def _validate_task_parallel_spec(task: InferenceTask, spec: ExecutionGroupSpec) -> None:
        if (
            int(task.parallel_spec.tp) != int(spec.parallel_spec.tp)
            or int(task.parallel_spec.sp) != int(spec.parallel_spec.sp)
            or int(task.parallel_spec.cfg) != int(spec.parallel_spec.cfg)
            or bool(task.parallel_spec.cfg_parallel) != bool(spec.parallel_spec.cfg_parallel)
        ):
            raise ValueError(
                f"task {task.task_id} parallel_spec does not match group {spec.group_id!r}: "
                f"task={task.parallel_spec!r} group={spec.parallel_spec!r}"
            )

    def _activate_group_session(self, group_id: str) -> None:
        session = self.fixed_sessions.get(str(group_id))
        if session is not None:
            session.activate()
        elif self.fixed_session is not None:
            self.fixed_session.activate()
        else:
            raise RuntimeError("runtime_v2 worker session is not initialized")
        self._apply_group_parallel_config(str(group_id))

    def _maybe_activate_group_session(self, group_id: str) -> None:
        if self.fixed_sessions or self.fixed_session is not None:
            self._activate_group_session(group_id)
            return
        self._apply_group_parallel_config(str(group_id))

    def _apply_group_parallel_config(self, group_id: str) -> None:
        group_id = str(group_id)
        # Fast path: the model tree already holds this group's parallel_config.
        # Re-walking transformer.modules() (thousands of submodules) on every
        # dit step is pure overhead — measured ~6 ms/step, exposed on the
        # critical path and paid even by static lanes that never change group.
        # session.activate() (cheap global pointer swap) still runs every step
        # in the caller, so global parallel state stays correct regardless.
        if group_id == self._active_parallel_group_id:
            return
        parallel_config = self._parallel_config_by_group.get(group_id)
        if parallel_config is None:
            return
        self.od_config.parallel_config = parallel_config
        if self.worker_wrapper is None:
            return
        worker = getattr(self.worker_wrapper, "worker", None)
        if worker is None:
            return
        worker_od_config = getattr(worker, "od_config", None)
        if worker_od_config is not None:
            worker_od_config.parallel_config = parallel_config
        pipeline = getattr(getattr(worker, "model_runner", None), "pipeline", None)
        if pipeline is None:
            return
        if hasattr(pipeline, "parallel_config"):
            pipeline.parallel_config = parallel_config
        pipeline_od_config = getattr(pipeline, "od_config", None)
        if pipeline_od_config is not None:
            pipeline_od_config.parallel_config = parallel_config
        # Enumerate parallel-aware modules once, then reuse: the module.modules()
        # recursion + hasattr scan over the 20B DiT is the expensive part. With
        # the fast-path guard above this only runs on real group changes, and
        # caching makes even those cheap (just setattr over the known targets).
        targets = self._pcfg_target_modules
        if targets is None:
            targets = []
            for attr in ("transformer", "transformer_2"):
                module = getattr(pipeline, attr, None)
                if module is None:
                    continue
                if hasattr(module, "parallel_config"):
                    targets.append(module)
                modules_iter = module.modules() if hasattr(module, "modules") else ()
                for submodule in modules_iter:
                    if hasattr(submodule, "parallel_config"):
                        targets.append(submodule)
            self._pcfg_target_modules = targets
        for module in targets:
            module.parallel_config = parallel_config
        # Record the fully-applied group so subsequent same-group steps skip
        # the walk. Set only after a complete apply (the early returns above
        # run before the pipeline exists, during init, and must not poison it).
        self._active_parallel_group_id = group_id

    @staticmethod
    def _request_metadata_for_task(task: InferenceTask) -> Mapping[str, Any]:
        metadata = task.payload.get("request_metadata") if isinstance(task.payload, Mapping) else None
        if isinstance(metadata, Mapping):
            return metadata
        return {}

    def _start_auto_profiler_if_enabled(self) -> None:
        if self._fetch_window_profile_enabled:
            return
        if self._auto_profile_first_n_reqs <= 0 or self._auto_profile_started:
            return
        trace_dir = os.path.abspath(os.path.expanduser(os.environ.get("VLLM_TORCH_PROFILER_DIR", "./profiles")))
        os.makedirs(trace_dir, exist_ok=True)
        trace_template = os.path.join(
            trace_dir,
            f"{self._auto_profile_trace_prefix}_{int(time.time())}_pid{os.getpid()}",
        )
        try:
            CurrentProfiler.start(trace_template)
        except Exception as exc:
            logger.warning(
                "runtime_v2 worker auto profiler start failed: rank=%s first_n_reqs=%s template=%s error=%s",
                self.worker_rank,
                self._auto_profile_first_n_reqs,
                trace_template,
                exc,
            )
            return
        self._auto_profile_started = True
        self._auto_profile_template = trace_template
        logger.info(
            "runtime_v2 worker auto profiler started: rank=%s first_n_reqs=%s template=%s",
            self.worker_rank,
            self._auto_profile_first_n_reqs,
            trace_template,
        )

    def _stop_auto_profiler(self, *, reason: str) -> None:
        if not self._auto_profile_started or self._auto_profile_stopped:
            return
        stop_begin_ns = time.monotonic_ns()
        try:
            result = CurrentProfiler.stop()
            stop_done_ns = time.monotonic_ns()
            logger.info(
                "runtime_v2 worker auto profiler stopped: rank=%s reason=%s completed_reqs=%s "
                "elapsed_ns=%s mono_ns=%s result=%s",
                self.worker_rank,
                reason,
                len(self._auto_profile_completed_requests),
                max(0, stop_done_ns - stop_begin_ns),
                stop_done_ns,
                result,
            )
        except Exception as exc:
            stop_done_ns = time.monotonic_ns()
            logger.warning(
                "runtime_v2 worker auto profiler stop failed: rank=%s reason=%s completed_reqs=%s "
                "elapsed_ns=%s mono_ns=%s error=%s",
                self.worker_rank,
                reason,
                len(self._auto_profile_completed_requests),
                max(0, stop_done_ns - stop_begin_ns),
                stop_done_ns,
                exc,
            )
        self._auto_profile_stopped = True

    def _maybe_stop_auto_profiler_after_request(self, request_id: str) -> None:
        if not self._auto_profile_started or self._auto_profile_stopped:
            return
        if request_id in self._auto_profile_completed_requests:
            return
        self._auto_profile_completed_requests.add(request_id)
        completed = len(self._auto_profile_completed_requests)
        logger.info(
            "runtime_v2 worker auto profiler progress: rank=%s completed_reqs=%s target_reqs=%s request_id=%s",
            self.worker_rank,
            completed,
            self._auto_profile_first_n_reqs,
            request_id,
        )
        if completed >= self._auto_profile_first_n_reqs:
            self._stop_auto_profiler(reason=f"first_{self._auto_profile_first_n_reqs}_requests")

    def _maybe_start_fetch_window_profiler(self, command: FetchArtifactsCommand) -> None:
        if not self._fetch_window_profile_enabled:
            return
        if self.worker_rank != 0:
            return
        if self._fetch_window_profile_active:
            return
        if self._fetch_window_profile_dispatch_target <= 0:
            return
        if (
            self._fetch_window_profile_target_request_id is not None
            and command.request_id != self._fetch_window_profile_target_request_id
        ):
            return

        trace_dir = os.path.abspath(os.path.expanduser(os.environ.get("VLLM_TORCH_PROFILER_DIR", "./profiles")))
        os.makedirs(trace_dir, exist_ok=True)
        self._fetch_window_profile_seq += 1
        safe_request_id = command.request_id.replace("/", "_")
        trace_template = os.path.join(
            trace_dir,
            f"{self._auto_profile_trace_prefix}_fetchwin_rank{self.worker_rank}_seq{self._fetch_window_profile_seq}"
            f"_req_{safe_request_id}_{int(time.time())}",
        )
        try:
            CurrentProfiler.start(trace_template)
        except Exception as exc:
            logger.warning(
                "runtime_v2 worker fetch-window profiler start failed: rank=%s request_id=%s dispatch_target=%s "
                "template=%s error=%s",
                self.worker_rank,
                command.request_id,
                self._fetch_window_profile_dispatch_target,
                trace_template,
                exc,
            )
            return

        self._fetch_window_profile_active = True
        self._fetch_window_profile_start_ns = time.monotonic_ns()
        self._fetch_window_profile_request_id = command.request_id
        self._fetch_window_profile_remaining_dispatches = self._fetch_window_profile_dispatch_target
        logger.info(
            "runtime_v2 worker fetch-window profiler started: rank=%s request_id=%s dispatch_target=%s template=%s",
            self.worker_rank,
            command.request_id,
            self._fetch_window_profile_dispatch_target,
            trace_template,
        )

    def _stop_fetch_window_profiler(self, *, reason: str) -> None:
        if not self._fetch_window_profile_active:
            return
        stop_begin_ns = time.monotonic_ns()
        window_elapsed_ns = (
            max(0, stop_begin_ns - self._fetch_window_profile_start_ns)
            if self._fetch_window_profile_start_ns > 0
            else 0
        )
        try:
            result = CurrentProfiler.stop()
            stop_done_ns = time.monotonic_ns()
            logger.info(
                "runtime_v2 worker fetch-window profiler stopped: rank=%s reason=%s trigger_request_id=%s "
                "window_elapsed_ns=%s stop_elapsed_ns=%s mono_ns=%s result=%s",
                self.worker_rank,
                reason,
                self._fetch_window_profile_request_id,
                window_elapsed_ns,
                max(0, stop_done_ns - stop_begin_ns),
                stop_done_ns,
                result,
            )
        except Exception as exc:
            stop_done_ns = time.monotonic_ns()
            logger.warning(
                "runtime_v2 worker fetch-window profiler stop failed: rank=%s reason=%s trigger_request_id=%s "
                "window_elapsed_ns=%s stop_elapsed_ns=%s mono_ns=%s error=%s",
                self.worker_rank,
                reason,
                self._fetch_window_profile_request_id,
                window_elapsed_ns,
                max(0, stop_done_ns - stop_begin_ns),
                stop_done_ns,
                exc,
            )
        self._fetch_window_profile_active = False
        self._fetch_window_profile_start_ns = 0
        self._fetch_window_profile_request_id = None
        self._fetch_window_profile_remaining_dispatches = 0

    def _advance_fetch_window_after_dispatch(self, command: ProcessDispatchTaskCommand) -> None:
        if not self._fetch_window_profile_active:
            return
        self._fetch_window_profile_remaining_dispatches = max(0, self._fetch_window_profile_remaining_dispatches - 1)
        logger.info(
            "runtime_v2 worker fetch-window profiler progress: rank=%s trigger_request_id=%s "
            "dispatch_task_id=%s dispatch_request_id=%s remaining_dispatches=%s",
            self.worker_rank,
            self._fetch_window_profile_request_id,
            command.task.task_id,
            command.task.request_id,
            self._fetch_window_profile_remaining_dispatches,
        )
        if self._fetch_window_profile_remaining_dispatches <= 0:
            self._stop_fetch_window_profiler(reason="dispatch_window_completed")

    def _prepare_fetch_copy_stream_dependency(self, command: FetchArtifactsCommand) -> None:
        if self._fetch_copy_stream is None or not torch.cuda.is_available():
            return
        wait_begin_ns = time.monotonic_ns()
        producer_stream = torch.cuda.current_stream()
        self._fetch_copy_stream.wait_stream(producer_stream)
        wait_done_ns = time.monotonic_ns()
        logger.debug(
            "runtime_v2 worker fetch stream wait: rank=%s request_id=%s fetch_id=%s "
            "elapsed_ns=%s mono_ns=%s",
            self.worker_rank,
            command.request_id,
            command.fetch_id,
            max(0, wait_done_ns - wait_begin_ns),
            wait_done_ns,
        )

    def run(self) -> None:
        try:
            self._initialize()
            self._start_fetch_thread()
            self.event_pipe_w.send(WorkerReadyMessage(worker_rank=self.worker_rank))
            while True:
                self._drain_dit_timing()
                recv_wait_begin_ns = time.monotonic_ns()
                command = self.command_pipe_r.recv()
                recv_ns = time.monotonic_ns()
                recv_wait_ns = max(0, recv_ns - recv_wait_begin_ns)
                logger.debug(
                    "runtime_v2 worker command recv: rank=%s cmd=%s wait_ns=%s mono_ns=%s",
                    self.worker_rank,
                    self._command_name(command),
                    recv_wait_ns,
                    recv_ns,
                )
                if isinstance(command, ShutdownWorkerCommand):
                    break
                if isinstance(command, ProcessDispatchTaskCommand):
                    self._execute_task(command)
                    self._advance_fetch_window_after_dispatch(command)
                    continue
                if isinstance(command, FetchArtifactsCommand):
                    self._prepare_fetch_copy_stream_dependency(command)
                    self._maybe_start_fetch_window_profiler(command)
                    self._fetch_queue.put(command)
                    continue
                if isinstance(command, MigrateArtifactsCommand):
                    self._ensure_group_spec(command.src_group)
                    self._ensure_group_spec(command.dst_group)
                    self._handle_migrate_artifacts(command)
                    continue
                if isinstance(command, EvictRequestCommand):
                    self._handle_evict_request(command)
                    continue
                if isinstance(command, RpcCommand):
                    raise RuntimeError("runtime_v2 multiproc worker rpc is temporarily disabled")
                raise TypeError(f"unsupported runtime_v2 worker command: {type(command)!r}")
        except Exception:
            self.event_pipe_w.send(
                WorkerReadyMessage(
                    worker_rank=self.worker_rank,
                    status="failed",
                    message=traceback.format_exc(),
                )
            )
            raise
        finally:
            self._shutdown()

    def _initialize(self) -> None:
        worker_config = _clone_runtime_worker_config(self.od_config)
        gpu_id = int(self.worker_rank if self.device_id is None else self.device_id)
        self.worker_wrapper = WorkerWrapperBase(
            gpu_id=gpu_id,
            local_rank=self.local_rank,
            rank=self.dist_rank,
            device_id=gpu_id,
            world_size=self.world_size,
            master_port=self.master_port,
            od_config=worker_config,
            worker_extension_cls=worker_config.worker_extension_cls,
            custom_pipeline_args=worker_config.custom_pipeline_args,
        )
        pipeline = self.worker_wrapper.worker.model_runner.pipeline
        if pipeline is None:
            raise RuntimeError(f"worker {self.worker_rank} failed to initialize diffusion pipeline")
        adapter = get_runtime_v2_adapter(getattr(self.od_config, "model_class_name", None))
        adapter.validate_pipeline(pipeline, self.od_config)
        self.executors = adapter.build_executors(pipeline)
        self._register_executor_codecs()
        self._execution_group_specs_by_id = self._build_execution_group_specs_by_id()
        self._parallel_config_by_group = self._build_parallel_configs_by_group()
        # Configs were just rebuilt (new objects); force the next activate to
        # re-apply them across the model tree rather than skip on a stale id,
        # and re-enumerate the cached parallel-aware module list.
        self._active_parallel_group_id = None
        self._pcfg_target_modules = None
        self.fixed_sessions = {}
        for group_id, spec in self._execution_group_specs_by_id.items():
            if self.worker_rank not in spec.ranks:
                continue
            omni_parallel_state.activate_runtime_v2_model_parallel_session(group_id)
            self.fixed_sessions[group_id] = _FixedParallelSession.capture_current(spec.parallel_spec)
        self.fixed_session = self.fixed_sessions.get(self.group_id)
        if self.fixed_session is None:
            self.fixed_session = _FixedParallelSession.capture_current(self.parallel_spec)
        self._activate_group_session(self.group_id)
        if torch.cuda.is_available():
            self._fetch_copy_stream = torch.cuda.Stream()
        else:
            self._fetch_copy_stream = None
        self._start_auto_profiler_if_enabled()
        logger.info(
            "runtime_v2 multiproc worker initialized: worker_rank=%s group=%s dist_rank=%s world_size=%s "
            "device_id=%s master_port=%s tp=%s sp=%s cfg=%s",
            self.worker_rank,
            self.group_id,
            self.dist_rank,
            self.world_size,
            gpu_id,
            self.master_port,
            self.parallel_spec.tp,
            self.parallel_spec.sp,
            self.parallel_spec.cfg,
        )

    def _register_executor_codecs(self) -> None:
        for executor in self.executors.values():
            for codec in executor.output_codecs.values():
                self.codec_registry.register(codec)

    def _shutdown(self) -> None:
        self._stop_fetch_thread()
        self._stop_fetch_window_profiler(reason="worker_shutdown")
        self._stop_auto_profiler(reason="worker_shutdown")
        if self.worker_wrapper is not None:
            try:
                self.worker_wrapper.shutdown()
            except Exception as exc:
                logger.warning("runtime_v2 worker shutdown failed: rank=%s error=%s", self.worker_rank, exc)

    def _start_fetch_thread(self) -> None:
        if self._fetch_thread is not None and self._fetch_thread.is_alive():
            return
        self._fetch_stop.clear()
        self._fetch_thread = threading.Thread(
            target=self._fetch_loop,
            name=f"runtime-v2-fetch-worker-{self.worker_rank}",
            daemon=True,
        )
        self._fetch_thread.start()

    def _stop_fetch_thread(self) -> None:
        self._fetch_stop.set()
        try:
            self._fetch_queue.put_nowait(None)
        except queue.Full:  # pragma: no cover - unbounded queue
            pass
        thread = self._fetch_thread
        if thread is not None:
            thread.join(timeout=5.0)
            self._fetch_thread = None

    def _fetch_loop(self) -> None:
        while not self._fetch_stop.is_set():
            try:
                command = self._fetch_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if command is None:
                return
            self._handle_fetch_artifacts(command)

    def _send_result(self, payload: Any) -> None:
        with self._result_pipe_lock:
            self.result_pipe_w.send(payload)

    def _resolve_inputs(
        self,
        task: InferenceTask,
        inline_inputs: tuple[SerializedArtifactValue, ...],
    ) -> dict[str, Any]:
        inline_by_id = {
            artifact_value.handle.artifact_id: _deserialize_artifact_value(artifact_value)
            for artifact_value in inline_inputs
        }
        resolved_inputs: dict[str, Any] = {}
        group_id = self._require_task_group_id(task)
        request_metadata = self._request_metadata_for_task(task)
        for artifact in task.inputs:
            key = self._artifact_key(artifact.request_id, group_id, artifact.artifact_id)
            with self._artifacts_lock:
                local_value = self.local_artifacts.get(key)
            if local_value is not None:
                self._materialize_input_artifact(
                    artifact=artifact,
                    value=local_value.value,
                    group_id=group_id,
                    request_metadata=request_metadata,
                )
                resolved_inputs[artifact.artifact_id] = local_value.value
                continue
            if artifact.artifact_id in inline_by_id:
                value = inline_by_id[artifact.artifact_id]
                self._materialize_input_artifact(
                    artifact=artifact,
                    value=value.value,
                    group_id=group_id,
                    request_metadata=request_metadata,
                )
                with self._artifacts_lock:
                    self.local_artifacts[key] = value
                resolved_inputs[artifact.artifact_id] = value.value
                continue
            raise KeyError(f"worker {self.worker_rank} cannot resolve input artifact {artifact.artifact_id}")
        return resolved_inputs

    def _make_event(
        self,
        task: InferenceTask,
        kind: WorkerEventKind,
        *,
        message: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> WorkerEvent:
        return WorkerEvent(
            event_id=str(uuid.uuid4()),
            task_id=task.task_id,
            request_id=task.request_id,
            group_id=self._require_task_group_id(task),
            worker_rank=self.worker_rank,
            kind=kind,
            timestamp_ns=time.monotonic_ns(),
            message=message,
            metadata=metadata or {},
        )

    def _drain_dit_timing(self) -> None:
        """Log GPU exec time for DiT chunks whose CUDA end event has completed.

        Polled with the non-blocking ``Event.query()`` — never synchronizes, so
        the dispatch loop is not serialized. A chunk's timing is logged a few
        task cycles after it ran, once its GPU work has drained. ``exec_gpu_ms``
        is the real on-device time; ``exec_cpu_ms`` is the (async, launch-bound)
        wall time the old ``exec_only_ms`` used to report.
        """
        if not self._pending_dit_timing:
            return
        still_pending: list[tuple[Any, ...]] = []
        for entry in self._pending_dit_timing:
            (
                task_id,
                group_id,
                cost_model_stage,
                do_true_cfg,
                gpu_start,
                gpu_end,
                activate_ms,
                cpu_ms,
                prev_dit_end,
                gpu_record_ns,
            ) = entry
            if not gpu_end.query():
                still_pending.append(entry)
                continue
            # GPU-stream idle between the previous DiT kernel's end and this
            # kernel's start (both CUDA events on the same stream). This is the
            # on-device bubble; -1 if no predecessor or not yet timeable.
            gpu_idle_ms = -1.0
            if prev_dit_end is not None:
                try:
                    gpu_idle_ms = prev_dit_end.elapsed_time(gpu_start)
                except Exception:
                    gpu_idle_ms = -1.0
            logger.info(
                "runtime_v2 worker dit chunk timing: rank=%s group=%s task_id=%s "
                "exec_gpu_ms=%.1f exec_cpu_ms=%.1f session_activate_ms=%.1f "
                "gpu_idle_ms=%.2f gpu_record_ns=%s "
                "cost_model_stage=%s do_true_cfg=%s",
                self.worker_rank,
                group_id,
                task_id,
                gpu_start.elapsed_time(gpu_end),
                cpu_ms,
                activate_ms,
                gpu_idle_ms,
                gpu_record_ns,
                cost_model_stage,
                do_true_cfg,
            )
        self._pending_dit_timing = still_pending

    def _execute_task(self, command: ProcessDispatchTaskCommand) -> None:
        if command.group_spec is not None:
            self._ensure_group_spec(command.group_spec)
        if self.fixed_session is None and not self.fixed_sessions:
            raise RuntimeError("runtime_v2 worker session is not initialized")
        if self.worker_wrapper is None:
            raise RuntimeError("runtime_v2 worker wrapper is not initialized")
        task = command.task
        # Local, rank-symmetric setup (group/spec validation, session activation,
        # executor lookup) is safe to downgrade to a per-task TASK_FAILED: every
        # rank of the group runs the same task, so a failure here happens
        # identically on all ranks and none is left waiting on a peer. This keeps
        # one malformed request from propagating through run() and killing the
        # whole worker process (the worker-init checks above stay fatal -- a
        # missing session/wrapper is not per-task recoverable).
        try:
            group_id = self._require_task_group_id(task)
            group_spec = command.group_spec if command.group_spec is not None else self._group_spec(group_id)
            self._validate_task_parallel_spec(task, group_spec)
            started_ns = time.monotonic_ns()
            queue_delay_ms = max(0.0, (started_ns - command.dispatched_at_ns) / 1_000_000) if command.dispatched_at_ns else 0.0
            _activate_begin_ns = time.monotonic_ns()
            self._activate_group_session(group_id)
            _activate_elapsed_ms = (time.monotonic_ns() - _activate_begin_ns) / 1_000_000
            if _activate_elapsed_ms > 50:
                logger.info(
                    "runtime_v2 worker slow session activate: rank=%s group=%s task_id=%s activate_ms=%.1f",
                    self.worker_rank,
                    group_id,
                    task.task_id,
                    _activate_elapsed_ms,
                )
            executor = self.executors.get(task.kind)
            if executor is None:
                self.event_pipe_w.send(
                    self._make_event(
                        task,
                        WorkerEventKind.TASK_FAILED,
                        message=f"worker does not support task kind {task.kind}",
                    )
                )
                return
        except Exception as exc:
            self.event_pipe_w.send(
                self._make_event(
                    task,
                    WorkerEventKind.TASK_FAILED,
                    message=f"task setup failed: {exc}\n{traceback.format_exc()}",
                )
            )
            return

        # Input resolution is deliberately NOT wrapped in the per-task TASK_FAILED
        # guard above: _resolve_inputs -> materialize_input_artifact issues a
        # cross-rank all_gather of sharded inputs, and a failure here is typically
        # asymmetric (e.g. an artifact present on some ranks but missing on
        # others). Catching it on the failing rank and continuing would leave the
        # peer ranks blocked forever inside that all_gather and desync this worker
        # for every later task. Letting the exception propagate to run() instead
        # tears the worker (and its process group) down, so the SP peers fail fast
        # rather than hang -- the pre-hardening behaviour for this path.
        resolved_inputs = self._resolve_inputs(task, command.inline_inputs)
        logger.debug(
            "runtime_v2 worker launch begin: rank=%s request_id=%s task_id=%s kind=%s "
            "queue_delay_ms=%.3f inline_inputs=%s mono_ns=%s dispatched_at_ns=%s",
            self.worker_rank,
            task.request_id,
            task.task_id,
            task.kind,
            queue_delay_ms,
            len(command.inline_inputs),
            started_ns,
            command.dispatched_at_ns,
        )
        self.event_pipe_w.send(
            self._make_event(
                task,
                WorkerEventKind.TASK_LAUNCH_BEGIN,
                metadata={
                    "queue_delay_ms": queue_delay_ms,
                    "worker_launch_begin_ns": started_ns,
                    "dispatched_at_ns": command.dispatched_at_ns,
                },
            )
        )
        self.event_pipe_w.send(self._make_event(task, WorkerEventKind.TASK_EXEC_BEGIN))
        try:
            worker = self.worker_wrapper.worker
            use_hsdp = bool(
                getattr(getattr(worker, "od_config", self.od_config), "parallel_config", None) is not None
                and getattr(getattr(worker, "od_config", self.od_config).parallel_config, "use_hsdp", False)
            )
            grad_context = torch.no_grad() if use_hsdp else torch.inference_mode()
            _exec_begin_ns = time.monotonic_ns()
            # GPU-accurate DiT chunk timing: bracket the exec with CUDA events
            # and read elapsed_time later from _drain_dit_timing once the end
            # event completes. CUDA is async, so the CPU monotonic delta below
            # only covers launch/dispatch, not GPU execution. Do NOT
            # synchronize here — that would serialize the dispatch pipeline.
            # Bracket EVERY stage (not just dit) with CUDA events on the compute
            # stream: prev_end.elapsed_time(this_start) then measures TRUE GPU-stream
            # idle between any two consecutive kernels (nothing but idle sits
            # between two kernels). cost_model_stage in the log tags the stage.
            _dit_timing = torch.cuda.is_available()
            _gpu_start = _gpu_end = None
            _gpu_record_ns = 0
            _prev_dit_end = None
            if _dit_timing:
                _gpu_start = torch.cuda.Event(enable_timing=True)
                _gpu_end = torch.cuda.Event(enable_timing=True)
                _gpu_start.record()
                _gpu_record_ns = time.monotonic_ns()
                _prev_dit_end = getattr(self, "_prev_dit_gpu_end", None)
            with set_forward_context(
                vllm_config=getattr(worker, "vllm_config", None),
                omni_diffusion_config=getattr(worker, "od_config", self.od_config),
            ):
                with grad_context:
                    outputs = executor.execute(task, resolved_inputs)
            if _dit_timing:
                _gpu_end.record()
                self._prev_dit_gpu_end = _gpu_end
                self._pending_dit_timing.append(
                    (
                        task.task_id,
                        group_id,
                        str(task.payload.get("cost_model_stage") or task.kind.value),
                        task.payload.get("do_true_cfg"),
                        _gpu_start,
                        _gpu_end,
                        _activate_elapsed_ms,
                        (time.monotonic_ns() - _exec_begin_ns) / 1_000_000,
                        _prev_dit_end,
                        _gpu_record_ns,
                    )
                )
            request_metadata = self._request_metadata_for_task(task)
            outputs = tuple(
                self._shard_output_artifact(
                    artifact_value=artifact_value,
                    group_id=group_id,
                    request_metadata=request_metadata,
                )
                for artifact_value in outputs
            )
            published_outputs: list[WorkerLocalArtifactRef] = []
            with self._artifacts_lock:
                for artifact_value in outputs:
                    key = self._artifact_key(
                        artifact_value.handle.request_id,
                        group_id,
                        artifact_value.handle.artifact_id,
                    )
                    self.local_artifacts[key] = artifact_value
                    if self.worker_rank == command.result_owner_rank:
                        published_outputs.append(
                            WorkerLocalArtifactRef(
                                handle=artifact_value.handle,
                                group_id=group_id,
                                worker_rank=self.worker_rank,
                            )
                        )
            self.event_pipe_w.send(
                self._make_event(
                    task,
                    WorkerEventKind.TASK_LAUNCH_END,
                    metadata={"published_outputs": tuple(published_outputs)},
                )
            )
            launch_end_ns = time.monotonic_ns()
            exec_elapsed_ns = max(0, launch_end_ns - started_ns)
            logger.debug(
                "runtime_v2 worker launch end: rank=%s request_id=%s task_id=%s kind=%s "
                "published_outputs=%s exec_elapsed=%.3fs mono_ns=%s exec_elapsed_ns=%s",
                self.worker_rank,
                task.request_id,
                task.task_id,
                task.kind,
                len(published_outputs),
                exec_elapsed_ns / 1_000_000_000.0,
                launch_end_ns,
                exec_elapsed_ns,
            )
            with self._artifacts_lock:
                for artifact_id in command.release_after_exec_artifact_ids:
                    self.local_artifacts.pop(self._artifact_key(task.request_id, group_id, artifact_id), None)
            self.event_pipe_w.send(self._make_event(task, WorkerEventKind.TASK_EXEC_END))
            done_ns = time.monotonic_ns()
            total_elapsed_ns = max(0, done_ns - started_ns)
            # In-place high-precision per-step timeline (env-gated). All monotonic_ns
            # captured locally on the worker -- nothing routed via the scheduler.
            # Join with the dit chunk timing line (exec_gpu_ms/gpu_idle_ms) by task_id.
            if _dit_timing and env_flag("RUNTIME_V2_STEPTRACE", False):
                logger.info(
                    "STEPTRACE rank=%s task_id=%s dispatched_ns=%s recv_ns=%s exec_begin_ns=%s "
                    "gpu_record_ns=%s launch_end_ns=%s done_ns=%s queue_delay_ms=%.3f activate_ms=%.3f",
                    self.worker_rank,
                    task.task_id,
                    command.dispatched_at_ns,
                    started_ns,
                    _exec_begin_ns,
                    _gpu_record_ns,
                    launch_end_ns,
                    done_ns,
                    queue_delay_ms,
                    _activate_elapsed_ms,
                )
            logger.debug(
                "runtime_v2 worker task done: rank=%s request_id=%s task_id=%s kind=%s "
                "outputs=%s elapsed=%.3fs mono_ns=%s total_elapsed_ns=%s",
                self.worker_rank,
                task.request_id,
                task.task_id,
                task.kind,
                len(published_outputs),
                total_elapsed_ns / 1_000_000_000.0,
                done_ns,
                total_elapsed_ns,
            )
            if task.kind == TaskKind.FINALIZE:
                self._maybe_stop_auto_profiler_after_request(task.request_id)
        except Exception as exc:
            self.event_pipe_w.send(
                self._make_event(
                    task,
                    WorkerEventKind.TASK_FAILED,
                    message=f"{exc}\n{traceback.format_exc()}",
                )
            )

    def _handle_fetch_artifacts(self, command: FetchArtifactsCommand) -> None:
        started_ns = time.monotonic_ns()
        # Fetch runs on a background thread. It must not mutate the process-wide
        # runtime_v2 parallel session while the main worker thread may be inside
        # model execution or an SP collective for a different group.
        copy_stream_used = self._fetch_copy_stream is not None
        logger.debug(
            "runtime_v2 worker op begin: rank=%s op=fetch_artifacts fetch_id=%s request_id=%s artifact_count=%s "
            "copy_stream_used=%s mono_ns=%s",
            self.worker_rank,
            command.fetch_id,
            command.request_id,
            len(command.artifact_ids),
            copy_stream_used,
            started_ns,
        )
        artifacts: list[SerializedArtifactValue] = []
        transport_counts: dict[str, int] = {}
        payload_bytes = 0
        serialize_begin_ns = started_ns
        stream_context = (
            torch.cuda.stream(self._fetch_copy_stream)
            if self._fetch_copy_stream is not None
            else contextlib.nullcontext()
        )
        with stream_context:
            for artifact_id in command.artifact_ids:
                key = self._artifact_key(command.request_id, command.group_id, artifact_id)
                with self._artifacts_lock:
                    value = self.local_artifacts.get(key)
                if value is None:
                    serialize_done_ns = time.monotonic_ns()
                    serialize_ns = max(0, serialize_done_ns - serialize_begin_ns)
                    send_result_begin_ns = serialize_done_ns
                    self._send_result(
                        FetchArtifactsResult(
                            fetch_id=command.fetch_id,
                            request_id=command.request_id,
                            worker_rank=self.worker_rank,
                            error=(
                                "artifact not found: "
                                f"request_id={command.request_id}, group_id={command.group_id}, "
                                f"artifact_id={artifact_id}"
                            ),
                        )
                    )
                    done_ns = time.monotonic_ns()
                    send_result_ns = max(0, done_ns - send_result_begin_ns)
                    elapsed_ns = max(0, done_ns - started_ns)
                    logger.debug(
                        "runtime_v2 worker op done: rank=%s op=fetch_artifacts fetch_id=%s request_id=%s status=error "
                        "artifact_count=%s transport=none payload_bytes=0 copy_stream_used=%s serialize_ns=%s "
                        "send_result_ns=%s elapsed_ns=%s mono_ns=%s",
                        self.worker_rank,
                        command.fetch_id,
                        command.request_id,
                        len(command.artifact_ids),
                        copy_stream_used,
                        serialize_ns,
                        send_result_ns,
                        elapsed_ns,
                        done_ns,
                    )
                    return
                serialized = _serialize_artifact_value(value, prefer_shm_output=True)
                payload_bytes += serialized.payload_nbytes if serialized.payload_nbytes > 0 else len(serialized.payload)
                transport_counts[serialized.transport] = transport_counts.get(serialized.transport, 0) + 1
                artifacts.append(serialized)
        serialize_done_ns = time.monotonic_ns()
        serialize_ns = max(0, serialize_done_ns - serialize_begin_ns)
        send_result_begin_ns = serialize_done_ns
        self._send_result(
            FetchArtifactsResult(
                fetch_id=command.fetch_id,
                request_id=command.request_id,
                worker_rank=self.worker_rank,
                artifacts=tuple(artifacts),
            )
        )
        done_ns = time.monotonic_ns()
        send_result_ns = max(0, done_ns - send_result_begin_ns)
        elapsed_ns = max(0, done_ns - started_ns)
        transport = "none"
        if len(transport_counts) == 1:
            transport = next(iter(transport_counts))
        elif len(transport_counts) > 1:
            transport = "mixed"
        logger.debug(
            "runtime_v2 worker op done: rank=%s op=fetch_artifacts fetch_id=%s request_id=%s status=ok "
            "artifact_count=%s transport=%s payload_bytes=%s copy_stream_used=%s serialize_ns=%s send_result_ns=%s "
            "elapsed_ns=%s mono_ns=%s",
            self.worker_rank,
            command.fetch_id,
            command.request_id,
            len(command.artifact_ids),
            transport,
            payload_bytes,
            copy_stream_used,
            serialize_ns,
            send_result_ns,
            elapsed_ns,
            done_ns,
        )

    def _handle_evict_request(self, command: EvictRequestCommand) -> None:
        started_ns = time.monotonic_ns()
        with self._artifacts_lock:
            keys = [key for key in self.local_artifacts if key[0] == command.request_id]
            for key in keys:
                self.local_artifacts.pop(key, None)
        done_ns = time.monotonic_ns()
        elapsed_ns = max(0, done_ns - started_ns)
        logger.debug(
            "runtime_v2 worker op done: rank=%s op=evict_request request_id=%s removed_keys=%s "
            "elapsed_ns=%s mono_ns=%s",
            self.worker_rank,
            command.request_id,
            len(keys),
            elapsed_ns,
            done_ns,
        )

    def _handle_rpc(self, command: RpcCommand) -> None:
        if self.worker_wrapper is None:
            raise RuntimeError("runtime_v2 worker wrapper is not initialized")
        started_ns = time.monotonic_ns()
        logger.debug(
            "runtime_v2 worker op begin: rank=%s op=rpc rpc_id=%s method=%s reply=%s argc=%s kwargc=%s mono_ns=%s",
            self.worker_rank,
            command.rpc_id,
            command.method,
            command.reply,
            len(command.args or ()),
            len(command.kwargs or {}),
            started_ns,
        )
        try:
            result = self.worker_wrapper.execute_method(command.method, *(command.args or ()), **(command.kwargs or {}))
            if command.reply:
                self.result_pipe_w.send(
                    RpcResult(rpc_id=command.rpc_id, worker_rank=self.worker_rank, result=result)
                )
            done_ns = time.monotonic_ns()
            elapsed_ns = max(0, done_ns - started_ns)
            logger.debug(
                "runtime_v2 worker op done: rank=%s op=rpc rpc_id=%s method=%s status=ok reply=%s "
                "elapsed_ns=%s mono_ns=%s",
                self.worker_rank,
                command.rpc_id,
                command.method,
                command.reply,
                elapsed_ns,
                done_ns,
            )
        except Exception as exc:
            if command.reply:
                self.result_pipe_w.send(
                    RpcResult(
                        rpc_id=command.rpc_id,
                        worker_rank=self.worker_rank,
                        error=f"{exc}\n{traceback.format_exc()}",
                    )
                )
            done_ns = time.monotonic_ns()
            elapsed_ns = max(0, done_ns - started_ns)
            logger.debug(
                "runtime_v2 worker op done: rank=%s op=rpc rpc_id=%s method=%s status=error reply=%s "
                "elapsed_ns=%s mono_ns=%s",
                self.worker_rank,
                command.rpc_id,
                command.method,
                command.reply,
                elapsed_ns,
                done_ns,
            )


def _worker_process_entrypoint(
    *,
    worker_rank: int,
    device_id: int | None,
    od_config: OmniDiffusionConfig,
    parallel_spec: ParallelSpec,
    dist_rank: int | None,
    local_rank: int | None,
    world_size: int | None,
    master_port: int | None,
    group_id: str,
    command_pipe_r: Connection,
    event_pipe_w: Connection,
    result_pipe_w: Connection,
) -> None:
    runtime = _WorkerProcessRuntime(
        worker_rank=worker_rank,
        device_id=device_id,
        od_config=od_config,
        parallel_spec=parallel_spec,
        dist_rank=dist_rank,
        local_rank=local_rank,
        world_size=world_size,
        master_port=master_port,
        group_id=group_id,
        command_pipe_r=command_pipe_r,
        event_pipe_w=event_pipe_w,
        result_pipe_w=result_pipe_w,
    )
    runtime.run()


@dataclass(frozen=True)
class WorkerProcessHandle:
    process: mp.Process
    worker_rank: int
    command_pipe_w: Connection
    event_pipe_r: Connection
    result_pipe_r: Connection


@dataclass
class _TaskDispatchState:
    group_id: str
    expected_ranks: frozenset[int]
    result_owner_rank: int
    events_by_kind: dict[WorkerEventKind, dict[int, WorkerEvent]] = field(default_factory=dict)


class MultiprocWorkerPool:
    """Centeralized multiprocess worker pool for diffusion runtime_v2."""

    def __init__(
        self,
        topology: RuntimeTopology,
        od_config: OmniDiffusionConfig,
    ) -> None:
        self.topology = topology
        self.od_config = od_config
        self._execution_groups_payload = self._serialize_execution_groups()
        self._mp_ctx = mp.get_context("spawn")
        self.worker_handles: dict[int, WorkerProcessHandle] = {}
        self._event_queue: queue.Queue[WorkerEvent | WorkerReadyMessage] = queue.Queue()
        # Result-channel demux: the reader thread is the single producer, but
        # there can be multiple consumers (migration drain via pump_migrations
        # on the scheduler tick, fetch drain from the API thread). Mixing both
        # types in one queue produces a race where each consumer can pop the
        # other's message and explode with "unexpected result type". Split into
        # typed queues so each consumer only ever sees its own type.
        self._result_queues: dict[int, queue.Queue[Any]] = {}
        self._migration_result_queues: dict[int, queue.Queue[Any]] = {}
        self._migration_engine = MigrationEngine(
            build_command=self._build_migration_command,
            send=self._send_migration_command,
        )
        self._task_dispatch_state: dict[str, _TaskDispatchState] = {}
        self._state_lock = threading.RLock()
        self._reader_stop = threading.Event()
        self._reader_error: BaseException | None = None
        self._reader_thread: threading.Thread | None = None
        self._fetch_lock = threading.RLock()
        self._inflight_fetches: dict[str, int] = {}
        self._completed_fetches: dict[str, FetchArtifactsResult] = {}

    @staticmethod
    def _thread_trace(action: str, *, elapsed_ns: int | None = None, **fields: Any) -> None:
        if elapsed_ns is not None and elapsed_ns < _CPU_THREAD_TRACE_MIN_NS:
            return
        thread = threading.current_thread()
        parts = [f"thread={thread.name}", f"tid={thread.ident}", f"action={action}"]
        if elapsed_ns is not None:
            parts.append(f"elapsed_ns={elapsed_ns}")
        parts.extend(f"{k}={v}" for k, v in fields.items())
        parts.append(f"mono_ns={time.monotonic_ns()}")
        logger.debug("runtime_v2 cpu thread trace: %s", " ".join(parts))

    def start(self, timeout_s: float = 600.0) -> None:
        if self.worker_handles:
            raise RuntimeError("runtime_v2 worker pool is already started")
        # Shared-world mode: all workers join one global torch.distributed world.
        # Group membership is represented by execution-group metadata, not by
        # separate world_size/master_port per group.
        global_world_size = len(self.topology.workers)
        shared_master_port = int(getattr(self.od_config, "master_port", 30005) or 30005)
        for worker in self.topology.workers:
            # Pick the largest SP group this worker belongs to as its bootstrap
            # config, so model-load time SP hooks are installed when any dynamic
            # step group needs them. Per-task execution still activates the
            # exact target group session before running the executor.
            group = max(
                self.topology.get_groups_for_worker(worker.worker_rank),
                key=lambda candidate: (
                    int(candidate.parallel_spec.sp),
                    int(candidate.parallel_spec.tp),
                    len(candidate.ranks),
                ),
            )
            group_world_size = len(group.ranks)
            dist_rank = worker.worker_rank
            local_rank = int(worker.device_id if worker.device_id is not None else worker.worker_rank)
            worker_od_config = self._build_worker_od_config(
                group_spec=group,
                group_world_size=group_world_size,
                global_world_size=global_world_size,
                shared_master_port=shared_master_port,
            )
            command_pipe_r, command_pipe_w = self._mp_ctx.Pipe(duplex=False)
            event_pipe_r, event_pipe_w = self._mp_ctx.Pipe(duplex=False)
            result_pipe_r, result_pipe_w = self._mp_ctx.Pipe(duplex=False)
            process = self._mp_ctx.Process(
                target=_worker_process_entrypoint,
                kwargs={
                    "worker_rank": worker.worker_rank,
                    "device_id": worker.device_id,
                    "od_config": worker_od_config,
                    "parallel_spec": group.parallel_spec,
                    "dist_rank": dist_rank,
                    "local_rank": local_rank,
                    "world_size": global_world_size,
                    "master_port": shared_master_port,
                    "group_id": group.group_id,
                    "command_pipe_r": command_pipe_r,
                    "event_pipe_w": event_pipe_w,
                    "result_pipe_w": result_pipe_w,
                },
                name=f"runtime-v2-worker-{worker.worker_rank}",
                daemon=True,
            )
            process.start()
            command_pipe_r.close()
            event_pipe_w.close()
            result_pipe_w.close()
            self.worker_handles[worker.worker_rank] = WorkerProcessHandle(
                process=process,
                worker_rank=worker.worker_rank,
                command_pipe_w=command_pipe_w,
                event_pipe_r=event_pipe_r,
                result_pipe_r=result_pipe_r,
            )
            self._result_queues[worker.worker_rank] = queue.Queue()
            self._migration_result_queues[worker.worker_rank] = queue.Queue()

        ready_workers: set[int] = set()
        deadline = time.monotonic() + timeout_s
        while len(ready_workers) < len(self.worker_handles):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                dead = {
                    rank: handle.process.exitcode
                    for rank, handle in self.worker_handles.items()
                    if not handle.process.is_alive()
                }
                self.shutdown()
                raise TimeoutError(f"timed out waiting for runtime_v2 workers to start; dead={dead}")
            ready = wait([handle.event_pipe_r for handle in self.worker_handles.values()], timeout=min(0.1, remaining))
            for reader in ready:
                event = reader.recv()
                if isinstance(event, WorkerReadyMessage):
                    if event.status != "ready":
                        self.shutdown()
                        raise RuntimeError(
                            f"runtime_v2 worker {event.worker_rank} failed to start: {event.message}"
                        )
                    ready_workers.add(event.worker_rank)
                else:
                    self._enqueue_event(event)
        self._reader_stop.clear()
        self._reader_error = None
        self._reader_thread = threading.Thread(
            target=self._reader_loop,
            name="runtime-v2-multiproc-forwarder",
            daemon=True,
        )
        self._reader_thread.start()
        logger.info("runtime_v2 multiproc worker pool started: workers=%s", sorted(self.worker_handles))

    def _build_worker_od_config(
        self,
        *,
        group_spec: ExecutionGroupSpec,
        group_world_size: int,
        global_world_size: int,
        shared_master_port: int,
    ) -> OmniDiffusionConfig:
        parallel_spec = group_spec.parallel_spec
        worker_config = _clone_runtime_worker_config(od_config=self.od_config)
        base_parallel = worker_config.parallel_config
        # Keep per-group parallel semantics for execution while still joining a
        # shared global world. This config is consumed by the runtime_v2 shared
        # initialization path in DiffusionWorker.
        worker_config.parallel_config = DiffusionParallelConfig(
            pipeline_parallel_size=1,
            data_parallel_size=1,
            tensor_parallel_size=int(parallel_spec.tp),
            enable_expert_parallel=bool(getattr(base_parallel, "enable_expert_parallel", False)),
            sequence_parallel_size=int(parallel_spec.sp),
            ulysses_degree=int(group_spec.ulysses_degree),
            ring_degree=int(group_spec.ring_degree),
            cfg_parallel_size=int(parallel_spec.cfg),
            vae_patch_parallel_size=int(getattr(base_parallel, "vae_patch_parallel_size", 1)),
            use_hsdp=False,
            hsdp_shard_size=-1,
            hsdp_replicate_size=1,
        )
        expected_world_size = int(worker_config.parallel_config.world_size)
        if expected_world_size != int(group_world_size):
            raise ValueError(
                f"invalid runtime_v2 group parallel spec: tp*sp*cfg={expected_world_size} "
                f"!= group_world_size={group_world_size}"
            )
        # Distributed bootstrap params are global in shared-world mode.
        worker_config.num_gpus = int(global_world_size)
        worker_config.master_port = int(shared_master_port)
        # Use a shared global distributed world; execution groups are realized
        # as sub-groups inside this world.
        setattr(worker_config, "runtime_v2_shared_world", True)
        setattr(worker_config, "runtime_v2_execution_groups", copy.deepcopy(self._execution_groups_payload))
        setattr(worker_config, "runtime_v2_worker_group_id", str(group_spec.group_id))
        return worker_config

    def _serialize_execution_groups(self) -> list[dict[str, Any]]:
        # Serialize runtime topology into a plain payload so worker subprocesses
        # can reconstruct model-parallel sub-groups deterministically.
        groups: list[dict[str, Any]] = []
        for group in self.topology.groups:
            groups.append(
                {
                    "group_id": str(group.group_id),
                    "ranks": [int(rank) for rank in group.ranks],
                    "tp": int(group.parallel_spec.tp),
                    "sp": int(group.parallel_spec.sp),
                    "cfg": int(group.parallel_spec.cfg),
                    "ulysses_degree": int(group.ulysses_degree),
                    "ring_degree": int(group.ring_degree),
                }
            )
        return groups

    def shutdown(self) -> None:
        self._reader_stop.set()
        for handle in self.worker_handles.values():
            try:
                handle.command_pipe_w.send(ShutdownWorkerCommand())
            except Exception:
                pass
        reader_thread = self._reader_thread
        if reader_thread is not None:
            reader_thread.join(timeout=1.0)
            self._reader_thread = None
        for handle in self.worker_handles.values():
            try:
                handle.process.join(timeout=5.0)
            except Exception:
                pass
            if handle.process.is_alive():
                handle.process.terminate()
                handle.process.join(timeout=5.0)
        self.worker_handles.clear()
        with self._state_lock:
            self._task_dispatch_state.clear()
        self._result_queues.clear()
        self._migration_result_queues.clear()
        with self._fetch_lock:
            self._inflight_fetches.clear()
            self._completed_fetches.clear()
        logger.info("runtime_v2 multiproc worker pool stopped")

    def dispatch(
        self,
        task: InferenceTask,
        inline_inputs: tuple[ArtifactValue, ...],
        release_after_exec_artifact_ids: tuple[str, ...] = (),
    ) -> None:
        if task.group_id is None:
            raise ValueError("task must have an assigned group before dispatch")
        group = self.topology.get_group(task.group_id)
        with self._state_lock:
            self._register_task_dispatch(task)
        command = ProcessDispatchTaskCommand(
            task=task,
            inline_inputs=tuple(_serialize_artifact_value(artifact_value) for artifact_value in inline_inputs),
            result_owner_rank=self.topology.get_group_leader(task.group_id),
            release_after_exec_artifact_ids=release_after_exec_artifact_ids,
            dispatched_at_ns=time.monotonic_ns(),
            group_spec=group,
        )
        for worker_rank in group.ranks:
            self.worker_handles[worker_rank].command_pipe_w.send(command)

    def poll(self, timeout_s: float = 0.0) -> list[WorkerEvent | WorkerReadyMessage]:
        self._raise_reader_error()
        events: list[WorkerEvent | WorkerReadyMessage] = []
        try:
            if timeout_s > 0:
                first = self._event_queue.get(timeout=timeout_s)
            else:
                first = self._event_queue.get_nowait()
        except queue.Empty:
            self._raise_reader_error()
            return events

        events.append(first)
        while True:
            try:
                events.append(self._event_queue.get_nowait())
            except queue.Empty:
                self._raise_reader_error()
                return events

    def fetch_artifacts(self, request_id: str, group_id: str, artifact_ids: tuple[str, ...]) -> FetchArtifactsResult:
        fetch_id = self.start_fetch_artifacts(request_id=request_id, group_id=group_id, artifact_ids=artifact_ids)
        deadline = time.monotonic() + 30.0
        while True:
            result = self.poll_fetch_artifacts(fetch_id)
            if result is not None:
                return self._normalize_fetch_result(result)
            if time.monotonic() >= deadline:
                self.discard_fetch(fetch_id)
                raise TimeoutError(f"timed out waiting for FetchArtifactsResult fetch_id={fetch_id}")
            time.sleep(0.001)

    def start_fetch_artifacts(self, request_id: str, group_id: str, artifact_ids: tuple[str, ...]) -> str:
        leader_rank = self.topology.get_group_leader(group_id)
        handle = self.worker_handles[leader_rank]
        fetch_id = str(uuid.uuid4())
        with self._fetch_lock:
            self._inflight_fetches[fetch_id] = leader_rank
        handle.command_pipe_w.send(
            FetchArtifactsCommand(
                fetch_id=fetch_id,
                request_id=request_id,
                group_id=group_id,
                artifact_ids=artifact_ids,
            )
        )
        return fetch_id

    def poll_fetch_artifacts(self, fetch_id: str) -> FetchArtifactsResult | None:
        with self._fetch_lock:
            completed = self._completed_fetches.pop(fetch_id, None)
            if completed is not None:
                self._inflight_fetches.pop(fetch_id, None)
                return self._normalize_fetch_result(completed)
            leader_rank = self._inflight_fetches.get(fetch_id)
            if leader_rank is None:
                return None

        self._drain_fetch_results_for_rank(leader_rank)
        with self._fetch_lock:
            completed = self._completed_fetches.pop(fetch_id, None)
            if completed is None:
                return None
            self._inflight_fetches.pop(fetch_id, None)
            return self._normalize_fetch_result(completed)

    def discard_fetch(self, fetch_id: str) -> None:
        with self._fetch_lock:
            self._inflight_fetches.pop(fetch_id, None)
            self._completed_fetches.pop(fetch_id, None)

    def begin_migration(
        self,
        plan: MigrateArtifactsPlan,
        *,
        on_done: Callable[[ArtifactMigrationSchedule | None], None],
        on_error: Callable[[Exception], None],
        timeout_s: float | None = None,
    ) -> str:
        """Start an asynchronous migration. Returns immediately with a
        migrate_id; the migration advances via pump_migrations() and invokes
        on_done(schedule) on success or on_error(exc) on failure/timeout."""
        self._raise_reader_error()
        src_group = self.topology.get_group(plan.src_group_id)
        dst_group = self.topology.get_group(plan.dst_group_id)
        participant_ranks = tuple(sorted(set(src_group.ranks) | set(dst_group.ranks)))
        for rank in participant_ranks:
            if rank not in self.worker_handles:
                raise RuntimeError(f"migration participant rank {rank} has no worker handle")
        for artifact in plan.artifacts:
            if not artifact.handle.codec_id:
                raise ValueError(
                    f"migration artifact {artifact.handle.artifact_id} has no codec_id")
        timeout = 30.0 if timeout_s is None else float(timeout_s)
        return self._migration_engine.begin(
            plan=plan,
            src_group=src_group,
            dst_group=dst_group,
            participant_ranks=participant_ranks,
            src_leader_rank=src_group.ranks[0],
            on_done=on_done,
            on_error=on_error,
            deadline=time.monotonic() + timeout,
        )

    def has_pending_migrations(self) -> bool:
        """True while any migration is in flight (used by the scheduler to cap its
        poll wait, since migration results do not wake poll())."""
        return self._migration_engine.has_pending()

    def pump_migrations(self) -> None:
        """Drain migration result queues (non-blocking) and advance all in-flight
        migrations one step. Called once per scheduler poll tick."""
        self._raise_reader_error()
        results: list[MigrateArtifactsRankResult] = []
        for result_queue in self._migration_result_queues.values():
            while True:
                try:
                    results.append(result_queue.get_nowait())
                except queue.Empty:
                    break
        self._migration_engine.pump(results, time.monotonic())

    def _build_migration_command(self, job: MigrationJob, phase: str) -> MigrateArtifactsCommand:
        # One builder for all phases: MigrateArtifactsCommand is frozen and the
        # worker-side handler reads only the fields relevant to each phase, so
        # passing all of them (empty/None until the engine fills them in) is
        # equivalent to the old per-phase construction and keeps this single.
        return MigrateArtifactsCommand(
            migrate_id=job.migrate_id,
            phase=phase,
            plan=job.plan,
            src_group=job.src_group,
            dst_group=job.dst_group,
            layout_results=tuple(job.layout_results),
            schedule=job.schedule,
            metadata_payloads=job.metadata_payloads,
        )

    def _send_migration_command(self, command: MigrateArtifactsCommand, ranks: tuple[int, ...]) -> None:
        for rank in ranks:
            self.worker_handles[rank].command_pipe_w.send(command)

    def _drain_fetch_results_for_rank(self, leader_rank: int) -> None:
        result_queue = self._result_queues.get(leader_rank)
        if result_queue is None:
            return
        while True:
            try:
                payload = result_queue.get_nowait()
            except queue.Empty:
                return
            if not isinstance(payload, FetchArtifactsResult):
                raise RuntimeError(
                    f"unexpected result type from worker {leader_rank}: expected FetchArtifactsResult, "
                    f"got {type(payload).__name__}"
                )
            fetch_id = payload.fetch_id
            if not fetch_id:
                raise RuntimeError(f"received fetch result without fetch_id from worker {leader_rank}")
            with self._fetch_lock:
                if fetch_id in self._inflight_fetches or fetch_id in self._completed_fetches:
                    self._completed_fetches[fetch_id] = payload
                else:
                    logger.debug(
                        "runtime_v2 drop stale fetch result: fetch_id=%s leader_rank=%s",
                        fetch_id,
                        leader_rank,
                    )

    @staticmethod
    def _normalize_fetch_result(result: FetchArtifactsResult) -> FetchArtifactsResult:
        artifacts: list[ArtifactValue] = []
        for artifact in result.artifacts:
            if isinstance(artifact, ArtifactValue):
                artifacts.append(artifact)
            else:
                # Keep SHM handles packed through scheduler/control path; unpack at
                # the final postprocess site to avoid extra host copies/serialization.
                artifacts.append(_deserialize_artifact_value(artifact, unpack_shm=False))
        return FetchArtifactsResult(
            fetch_id=result.fetch_id,
            request_id=result.request_id,
            worker_rank=result.worker_rank,
            artifacts=tuple(artifacts),
            error=result.error,
        )

    def evict_request(self, request_id: str) -> None:
        command = EvictRequestCommand(request_id=request_id)
        for handle in self.worker_handles.values():
            handle.command_pipe_w.send(command)

    def collective_rpc(
        self,
        method: str,
        *,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
        target_ranks: tuple[int, ...] | None = None,
        timeout_s: float | None = None,
    ) -> list[Any]:
        raise NotImplementedError("runtime_v2 multiproc worker pool rpc is temporarily disabled")

    def check_health(self) -> None:
        for handle in self.worker_handles.values():
            if not handle.process.is_alive():
                raise RuntimeError(
                    f"runtime_v2 worker {handle.worker_rank} died unexpectedly with exit code {handle.process.exitcode}"
                )

    def _wait_for_result(
        self,
        *,
        leader_rank: int,
        timeout_s: float,
        expected_type: type,
    ) -> Any:
        deadline = time.monotonic() + timeout_s
        result_queue = self._result_queues[leader_rank]
        while True:
            self._raise_reader_error()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"timed out waiting for {expected_type.__name__} from worker {leader_rank}")
            try:
                result = result_queue.get(timeout=min(0.1, remaining))
            except queue.Empty:
                continue
            if isinstance(result, expected_type):
                return result
            raise RuntimeError(
                f"unexpected result type from worker {leader_rank}: expected {expected_type.__name__}, "
                f"got {type(result).__name__}"
            )

    def _raise_reader_error(self) -> None:
        if self._reader_error is not None:
            raise RuntimeError("runtime_v2 multiproc pipe forwarder failed") from self._reader_error

    def _reader_loop(self) -> None:
        try:
            while not self._reader_stop.is_set():
                readers = [
                    handle.event_pipe_r
                    for handle in self.worker_handles.values()
                ] + [
                    handle.result_pipe_r
                    for handle in self.worker_handles.values()
                ]
                if not readers:
                    return
                wait_begin_ns = time.monotonic_ns()
                ready = wait(readers, timeout=0.1)
                wait_done_ns = time.monotonic_ns()
                self._thread_trace(
                    "forwarder_wait",
                    elapsed_ns=max(0, wait_done_ns - wait_begin_ns),
                    ready_count=len(ready),
                    total_readers=len(readers),
                )
                if not ready:
                    continue
                for reader in ready:
                    worker_rank = self._find_reader_owner_rank(reader)
                    if worker_rank is None:
                        continue
                    handle = self.worker_handles.get(worker_rank)
                    if handle is None:
                        continue
                    try:
                        recv_begin_ns = time.monotonic_ns()
                        payload = reader.recv()
                        recv_done_ns = time.monotonic_ns()
                        self._thread_trace(
                            "forwarder_recv",
                            elapsed_ns=max(0, recv_done_ns - recv_begin_ns),
                            worker_rank=worker_rank,
                            reader_kind="event" if reader is handle.event_pipe_r else "result",
                            payload_type=type(payload).__name__,
                        )
                    except EOFError as exc:
                        if self._reader_stop.is_set():
                            return
                        self._reader_error = exc
                        return
                    if reader is handle.event_pipe_r:
                        self._enqueue_event(payload)
                    elif isinstance(payload, MigrateArtifactsRankResult):
                        # Migration results have a dedicated queue so the
                        # scheduler-thread collector never competes with the
                        # API-thread fetch drain (which reads from
                        # _result_queues). See the comment on _result_queues
                        # in __init__.
                        self._migration_result_queues[worker_rank].put(payload)
                    else:
                        self._result_queues[worker_rank].put(payload)
        except BaseException as exc:  # pragma: no cover - defensive path
            self._reader_error = exc
            logger.exception("runtime_v2 multiproc pipe forwarder failed")

    def _find_reader_owner_rank(self, reader: Connection) -> int | None:
        for worker_rank, handle in self.worker_handles.items():
            if reader is handle.event_pipe_r or reader is handle.result_pipe_r:
                return worker_rank
        return None

    def _register_task_dispatch(self, task: InferenceTask) -> None:
        if task.group_id is None:
            raise ValueError("task must have an assigned group before registration")
        group = self.topology.get_group(task.group_id)
        self._task_dispatch_state[task.task_id] = _TaskDispatchState(
            group_id=task.group_id,
            expected_ranks=frozenset(group.ranks),
            result_owner_rank=self.topology.get_group_leader(task.group_id),
        )

    def _consume_event(self, event: WorkerEvent | WorkerReadyMessage) -> list[WorkerEvent | WorkerReadyMessage]:
        if not isinstance(event, WorkerEvent):
            return [event]
        if event.kind in (
            WorkerEventKind.TASK_LAUNCH_BEGIN,
            WorkerEventKind.TASK_LAUNCH_END,
            WorkerEventKind.TASK_EXEC_BEGIN,
            WorkerEventKind.TASK_EXEC_END,
            WorkerEventKind.TASK_FAILED,
        ):
            with self._state_lock:
                return self._aggregate_task_event(event)
        return [event]

    def _enqueue_event(self, event: WorkerEvent | WorkerReadyMessage) -> None:
        for aggregated_event in self._consume_event(event):
            if isinstance(aggregated_event, WorkerEvent):
                logger.debug(
                    "runtime_v2 event enqueued: request_id=%s task_id=%s kind=%s group=%s worker_rank=%s metadata=%s",
                    aggregated_event.request_id,
                    aggregated_event.task_id,
                    aggregated_event.kind,
                    aggregated_event.group_id,
                    aggregated_event.worker_rank,
                    dict(aggregated_event.metadata),
                )
            self._event_queue.put(aggregated_event)

    def _aggregate_task_event(self, event: WorkerEvent) -> list[WorkerEvent]:
        state = self._task_dispatch_state.get(event.task_id)
        if state is None:
            return []

        if event.kind == WorkerEventKind.TASK_FAILED:
            self._task_dispatch_state.pop(event.task_id, None)
            return [
                WorkerEvent(
                    event_id=f"aggregate:{event.task_id}:{event.kind.value}",
                    task_id=event.task_id,
                    request_id=event.request_id,
                    group_id=state.group_id,
                    worker_rank=event.worker_rank,
                    kind=event.kind,
                    timestamp_ns=event.timestamp_ns,
                    message=event.message,
                    metadata={"failed_rank": event.worker_rank},
                )
            ]

        seen_by_rank = state.events_by_kind.setdefault(event.kind, {})
        if event.worker_rank in seen_by_rank:
            return []
        seen_by_rank[event.worker_rank] = event
        if frozenset(seen_by_rank) != state.expected_ranks:
            logger.debug(
                "runtime_v2 event pending aggregation: request_id=%s task_id=%s kind=%s seen_ranks=%s expected_ranks=%s",
                event.request_id,
                event.task_id,
                event.kind,
                tuple(sorted(seen_by_rank)),
                tuple(sorted(state.expected_ranks)),
            )
            return []

        aggregated_event = self._build_aggregated_event(
            state=state,
            kind=event.kind,
            events_by_rank=seen_by_rank,
        )
        logger.debug(
            "runtime_v2 event aggregated: request_id=%s task_id=%s kind=%s ranks=%s metadata=%s",
            aggregated_event.request_id,
            aggregated_event.task_id,
            aggregated_event.kind,
            tuple(sorted(seen_by_rank)),
            dict(aggregated_event.metadata),
        )
        if event.kind == WorkerEventKind.TASK_EXEC_END:
            self._task_dispatch_state.pop(event.task_id, None)
        return [aggregated_event]

    def _build_aggregated_event(
        self,
        *,
        state: _TaskDispatchState,
        kind: WorkerEventKind,
        events_by_rank: dict[int, WorkerEvent],
    ) -> WorkerEvent:
        owner_event = events_by_rank.get(state.result_owner_rank)
        representative = owner_event or min(
            events_by_rank.values(),
            key=lambda event: (event.timestamp_ns, event.worker_rank),
        )
        metadata = dict(owner_event.metadata) if owner_event is not None else dict(representative.metadata)
        metadata["completed_ranks"] = tuple(sorted(events_by_rank))
        return WorkerEvent(
            event_id=f"aggregate:{representative.task_id}:{kind.value}",
            task_id=representative.task_id,
            request_id=representative.request_id,
            group_id=state.group_id,
            worker_rank=state.result_owner_rank,
            kind=kind,
            timestamp_ns=max(event.timestamp_ns for event in events_by_rank.values()),
            message=representative.message,
            metadata=metadata,
        )
