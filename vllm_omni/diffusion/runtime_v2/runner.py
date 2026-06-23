# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any

from vllm.logger import init_logger

from vllm_omni.diffusion.runtime_v2.adapters.common import (
    STAGE_AUX,
    STAGE_DIT,
    STANDARD_TASK_KINDS,
)
from vllm_omni.diffusion.runtime_v2.interfaces import RuntimeV2Adapter, SchedulerPolicy
from vllm_omni.diffusion.runtime_v2.multiproc_worker import MultiprocWorkerPool
from vllm_omni.diffusion.runtime_v2.protocol import ExecutionGroupSpec, ParallelSpec, TaskKind
from vllm_omni.diffusion.runtime_v2.registry import get_runtime_v2_adapter
from vllm_omni.diffusion.runtime_v2.cost_model import CostModelStore
from vllm_omni.diffusion.runtime_v2.scheduler import (
    DisaggregateSchedulerPolicy,
    DynamicStepFCFSSchedulerPolicy,
    EdfBestFitSchedulerPolicy,
    EdfGreedySchedulerPolicy,
    FCFSSchedulerPolicy,
    GlobalScheduler,
    InMemoryArtifactStore,
    SRTFSchedulerPolicy,
    TaskRuntimeEstimator,
    WaveStressSchedulerPolicy,
)
from vllm_omni.diffusion.runtime_v2.topology import RuntimeTopology, WorkerSpec

logger = init_logger(__name__)


_COST_MODEL_FALLBACK_WARNED: set[tuple[Any, ...]] = set()


def _warn_cost_model_fallback(parallel_spec: Any, stage: Any) -> None:
    """Emit a one-time warning per (sp, stage) when the cost model has no entry.

    A missing entry makes the estimator return 0.0, which deadline-aware policies
    (edf_best_fit) read as "finishes instantly / deadline always met" and therefore
    silently collapse the request to the smallest SP. Logging once surfaces the
    incomplete cost model so the degradation is diagnosable instead of invisible.
    """
    key = (getattr(parallel_spec, "sp", None), getattr(parallel_spec, "cfg", None), str(stage))
    if key in _COST_MODEL_FALLBACK_WARNED:
        return
    _COST_MODEL_FALLBACK_WARNED.add(key)
    logger.warning(
        "runtime_v2 cost model has no entry for sp=%s cfg=%s stage=%s; estimator returns 0.0, "
        "so deadline-aware scheduling will degrade toward the smallest SP for this stage. "
        "Profile the missing SP/stage to restore cost-aware placement.",
        key[0], key[1], key[2],
    )


def _make_cost_model_estimator(store: "CostModelStore") -> TaskRuntimeEstimator:
    """Wrap a CostModelStore into the (parallel_spec, seq_len, stage) callable
    that scheduler policies expect.

    If no cost-model entry exists for the requested parallel_spec, returns 0.0 and
    logs a one-time warning (see _warn_cost_model_fallback). 0.0 keeps the
    scheduling arithmetic numerically stable (unlike inf, which would poison
    SRTF's remaining-work charging), but the warning makes the missing-data case
    visible so it is not mistaken for a genuinely instantaneous stage.
    """

    def estimator(parallel_spec, seq_len, stage):
        try:
            model = store.for_parallel_spec(parallel_spec)
        except KeyError:
            _warn_cost_model_fallback(parallel_spec, stage)
            return 0.0
        return model.estimate_ms(str(stage), float(seq_len))

    return estimator


class RuntimeV2Runner:
    """Bootstrap runtime_v2 runner using an explicit model adapter."""

    def __init__(
        self,
        pipeline: Any | None = None,
        default_step_chunk_size: int = 1,
        *,
        scheduler_policy: str | SchedulerPolicy = "fcfs",
        task_runtime_estimator: TaskRuntimeEstimator | None = None,
        vllm_config: Any = None,
        omni_diffusion_config: Any = None,
    ) -> None:
        self.default_step_chunk_size = default_step_chunk_size
        if omni_diffusion_config is None:
            omni_diffusion_config = getattr(pipeline, "od_config", None)
        if omni_diffusion_config is None:
            raise ValueError("runtime_v2 runner requires omni_diffusion_config")
        self.od_config = omni_diffusion_config
        self.adapter = self._resolve_adapter(omni_diffusion_config)
        if pipeline is not None:
            self.adapter.validate_pipeline(pipeline, omni_diffusion_config)
        topology = self._build_topology(omni_diffusion_config, default_task_kinds=self.adapter.supported_task_kinds)
        self._validate_collective_backend(omni_diffusion_config, topology)
        policy_name = (
            str(scheduler_policy).lower()
            if not isinstance(scheduler_policy, SchedulerPolicy)
            else type(scheduler_policy).__name__.lower()
        )
        if "edf" in policy_name and str(
            getattr(omni_diffusion_config, "runtime_v2_collective_backend", "torch") or "torch"
        ).lower() != "gfc":
            raise ValueError(
                f"runtime_v2 scheduler_policy={policy_name!r} requires runtime_v2_collective_backend='gfc'"
            )
        if ("wave_stress" in policy_name or "wavestress" in policy_name) and str(
            getattr(omni_diffusion_config, "runtime_v2_collective_backend", "torch") or "torch"
        ).lower() != "gfc":
            raise ValueError(
                "runtime_v2 scheduler_policy='wave_stress' requires runtime_v2_collective_backend='gfc'"
            )
        stage_group_ids = self._resolve_stage_group_ids(
            scheduler_policy=scheduler_policy,
            omni_diffusion_config=omni_diffusion_config,
            topology=topology,
        )
        worker_pool = MultiprocWorkerPool(
            topology=topology,
            od_config=omni_diffusion_config,
        )
        self.worker_pool = worker_pool
        # Auto-load a cost-model dir into an estimator for any policy that
        # needs end-to-end finish-time predictions (currently edf_best_fit).
        # Caller-supplied task_runtime_estimator wins over the auto-loaded one.
        cost_model_dir = getattr(omni_diffusion_config, "runtime_v2_cost_model_dir", None)
        if task_runtime_estimator is None and cost_model_dir:
            cost_store = CostModelStore.load_dir(str(cost_model_dir))
            task_runtime_estimator = _make_cost_model_estimator(cost_store)
            logger.info(
                "runtime_v2 loaded cost-model from %s (sp_sizes=%s)",
                cost_model_dir,
                cost_store.available_sp_sizes(),
            )
        policy = self._build_scheduler_policy(
            topology=topology,
            scheduler_policy=scheduler_policy,
            task_runtime_estimator=task_runtime_estimator,
            stage_group_ids=stage_group_ids,
            dit_step_schedule=tuple(getattr(omni_diffusion_config, "runtime_v2_dit_step_schedule", None) or ()),
            edf_greedy_sp_sizes=tuple(getattr(omni_diffusion_config, "runtime_v2_edf_greedy_sp_sizes", None) or ()),
            collective_backend=str(
                getattr(omni_diffusion_config, "runtime_v2_collective_backend", "torch") or "torch"
            ).lower(),
            wave_stress_warmup_reqs=int(
                getattr(omni_diffusion_config, "runtime_v2_wave_stress_warmup_reqs", 0) or 0
            ),
            wave_stress_wave_size=int(
                getattr(omni_diffusion_config, "runtime_v2_wave_stress_wave_size", 4) or 4
            ),
        )
        self.scheduler = GlobalScheduler(
            topology=topology,
            worker_pool=worker_pool,
            compiler=self.adapter.build_task_compiler(
                default_denoise_chunk_size=default_step_chunk_size,
                od_config=omni_diffusion_config,
                pipeline=pipeline,
            ),
            artifact_store=InMemoryArtifactStore(),
            policy=policy,
        )
        self.scheduler.start()
        self._poll_interval_s = 0.01
        logger.info(
            "runtime_v2 runner started: model_class_name=%s adapter=%s default_step_chunk_size=%s "
            "groups=%s workers=%s policy=%s",
            getattr(omni_diffusion_config, "model_class_name", None),
            type(self.adapter).__name__,
            default_step_chunk_size,
            len(topology.groups),
            len(topology.workers),
            type(self.scheduler.policy).__name__,
        )

    @staticmethod
    def _resolve_adapter(omni_diffusion_config: Any) -> RuntimeV2Adapter:
        return get_runtime_v2_adapter(getattr(omni_diffusion_config, "model_class_name", None))

    @staticmethod
    def _validate_collective_backend(omni_diffusion_config: Any, topology: RuntimeTopology) -> None:
        backend = str(getattr(omni_diffusion_config, "runtime_v2_collective_backend", "torch") or "torch").lower()
        if backend != "gfc":
            return
        for group in topology.groups:
            if int(group.parallel_spec.tp) != 1:
                raise ValueError(
                    "runtime_v2_collective_backend='gfc' does not replace tensor-parallel all_reduce in v1; "
                    f"group {group.group_id!r} must use tp=1"
                )
            if int(group.parallel_spec.cfg) != 1:
                raise ValueError(
                    "runtime_v2_collective_backend='gfc' does not replace CFG collectives in v1; "
                    f"group {group.group_id!r} must use cfg=1"
                )
            # GFC pins symmetric memory sized for max_group_size, which the worker
            # caps at min(world_size, 16) (see diffusion_worker._init_runtime_v2_
            # collective). A group larger than that would exceed the configured cap
            # and fail opaquely inside GFC register_group/the first collective, so
            # reject it here with a clear, actionable error. Lift this together with
            # the worker's max_group_size cap if GFC v-next supports wider groups.
            if len(group.ranks) > 16:
                raise ValueError(
                    "runtime_v2_collective_backend='gfc' supports execution groups of at most "
                    f"16 ranks in v1; group {group.group_id!r} has {len(group.ranks)} ranks"
                )

    @staticmethod
    def _build_scheduler_policy(
        *,
        topology: RuntimeTopology,
        scheduler_policy: str | SchedulerPolicy,
        task_runtime_estimator: TaskRuntimeEstimator | None,
        stage_group_ids: Mapping[str, str] | None = None,
        dit_step_schedule: tuple[Mapping[str, Any], ...] | None = None,
        edf_greedy_sp_sizes: tuple[int, ...] | None = None,
        collective_backend: str = "torch",
        wave_stress_warmup_reqs: int = 0,
        wave_stress_wave_size: int = 4,
    ) -> SchedulerPolicy:
        if isinstance(scheduler_policy, SchedulerPolicy):
            if task_runtime_estimator is not None:
                logger.warning(
                    "runtime_v2 runner received task_runtime_estimator but scheduler_policy is already an instance; "
                    "estimator will be ignored"
                )
            if isinstance(scheduler_policy, DisaggregateSchedulerPolicy):
                if stage_group_ids and getattr(scheduler_policy, "stage_group_ids", None) is None:
                    scheduler_policy.stage_group_ids = dict(stage_group_ids)
                if (
                    isinstance(scheduler_policy, DynamicStepFCFSSchedulerPolicy)
                    and dit_step_schedule
                    and not getattr(scheduler_policy, "dit_step_schedule", None)
                ):
                    scheduler_policy.dit_step_schedule = tuple(dict(item) for item in dit_step_schedule)
            return scheduler_policy

        policy_name = str(scheduler_policy).lower()
        if policy_name == "fcfs":
            return FCFSSchedulerPolicy(
                topology=topology,
                task_runtime_estimator=task_runtime_estimator,
            )
        if policy_name == "srtf":
            return SRTFSchedulerPolicy(
                topology=topology,
                task_runtime_estimator=task_runtime_estimator,
            )
        if policy_name == "disaggregate":
            if task_runtime_estimator is not None:
                logger.warning(
                    "runtime_v2 disaggregate policy ignores task_runtime_estimator (FCFS within each group)"
                )
            return DisaggregateSchedulerPolicy(
                topology=topology,
                stage_group_ids=dict(stage_group_ids) if stage_group_ids else None,
            )
        if policy_name == "dynamic_step_fcfs":
            bad_tp = [group.group_id for group in topology.groups if int(group.parallel_spec.tp) != 1]
            if bad_tp:
                raise ValueError(
                    "runtime_v2 dynamic_step_fcfs currently supports only tp=1 execution groups; "
                    f"got non-tp1 groups: {bad_tp!r}"
                )
            if task_runtime_estimator is not None:
                logger.warning(
                    "runtime_v2 dynamic_step_fcfs policy ignores task_runtime_estimator (FCFS within each group)"
                )
            return DynamicStepFCFSSchedulerPolicy(
                topology=topology,
                stage_group_ids=dict(stage_group_ids) if stage_group_ids else None,
                dit_step_schedule=tuple(dict(item) for item in dit_step_schedule or ()),
            )
        if policy_name == "edf_greedy":
            bad_tp = [group.group_id for group in topology.groups if int(group.parallel_spec.tp) != 1]
            if bad_tp:
                raise ValueError(
                    "runtime_v2 edf_greedy currently supports only tp=1 execution groups; "
                    f"got non-tp1 groups: {bad_tp!r}"
                )
            return EdfGreedySchedulerPolicy(
                topology=topology,
                task_runtime_estimator=task_runtime_estimator,
                allowed_sp_sizes=edf_greedy_sp_sizes or None,
            )
        if policy_name == "edf_best_fit":
            bad_tp = [group.group_id for group in topology.groups if int(group.parallel_spec.tp) != 1]
            if bad_tp:
                raise ValueError(
                    "runtime_v2 edf_best_fit currently supports only tp=1 execution groups; "
                    f"got non-tp1 groups: {bad_tp!r}"
                )
            if task_runtime_estimator is None:
                logger.warning(
                    "runtime_v2 edf_best_fit has no task_runtime_estimator (no "
                    "cost-model dir configured); policy will degrade to "
                    "largest-fit until --runtime-v2-cost-model-dir is provided"
                )
            return EdfBestFitSchedulerPolicy(
                topology=topology,
                task_runtime_estimator=task_runtime_estimator,
                allowed_sp_sizes=edf_greedy_sp_sizes or None,
            )
        if policy_name == "wave_stress":
            bad_tp = [group.group_id for group in topology.groups if int(group.parallel_spec.tp) != 1]
            if bad_tp:
                raise ValueError(
                    "runtime_v2 wave_stress currently supports only tp=1 execution groups; "
                    f"got non-tp1 groups: {bad_tp!r}"
                )
            return WaveStressSchedulerPolicy(
                topology=topology,
                warmup_reqs=int(wave_stress_warmup_reqs),
                wave_size=int(wave_stress_wave_size),
                stage_group_ids=dict(stage_group_ids) if stage_group_ids else None,
                dit_step_schedule=tuple(dict(item) for item in dit_step_schedule or ()) or None,
                collective_backend=collective_backend,
            )
        raise ValueError(
            "runtime_v2 scheduler_policy must be one of "
            f"('fcfs', 'srtf', 'disaggregate', 'dynamic_step_fcfs', 'edf_greedy', 'edf_best_fit', 'wave_stress'), "
            f"got {scheduler_policy!r}"
        )

    @classmethod
    def _resolve_stage_group_ids(
        cls,
        *,
        scheduler_policy: str | SchedulerPolicy,
        omni_diffusion_config: Any,
        topology: RuntimeTopology,
    ) -> dict[str, str] | None:
        # disaggregate stage placement is opt-in only when:
        #   - scheduler policy is the "disaggregate" name (string) or instance
        #   - aux/dit group ids are both explicitly set on od_config
        # The latter is already enforced by OmniDiffusionConfig.__post_init__
        # for the string-name path (config validation refuses to construct a
        # disaggregate config without both ids); we re-check here for symmetry
        # so a programmatic policy instance also gets the same contract.
        if isinstance(scheduler_policy, SchedulerPolicy):
            policy_name = type(scheduler_policy).__name__
            if "Disaggregate" not in policy_name and "DynamicStep" not in policy_name:
                return None
        else:
            if str(scheduler_policy).lower() not in {"disaggregate", "dynamic_step_fcfs"}:
                return None
        aux_id = getattr(omni_diffusion_config, "runtime_v2_disaggregate_aux_group_id", None)
        dit_id = getattr(omni_diffusion_config, "runtime_v2_disaggregate_dit_group_id", None)
        if not aux_id or not dit_id:
            raise ValueError(
                "runtime_v2 scheduler_policy='disaggregate' requires both "
                "--runtime-v2-disaggregate-aux-group-id and "
                f"--runtime-v2-disaggregate-dit-group-id (got aux={aux_id!r}, dit={dit_id!r})"
            )
        if aux_id == dit_id:
            raise ValueError(
                "runtime_v2 disaggregate aux/dit group ids must differ; "
                f"got both = {aux_id!r}"
            )
        cls._validate_stage_group(topology, aux_id, STAGE_AUX)
        cls._validate_stage_group(topology, dit_id, STAGE_DIT)
        return {STAGE_AUX: aux_id, STAGE_DIT: dit_id}

    @staticmethod
    def _validate_stage_group(topology: RuntimeTopology, group_id: str, stage: str) -> None:
        try:
            group = topology.get_group(group_id)
        except KeyError as exc:
            raise ValueError(
                f"runtime_v2 disaggregate {stage} group_id={group_id!r} does not exist in topology; "
                f"available groups: {[g.group_id for g in topology.groups]!r}"
            ) from exc
        required_kinds = (
            (TaskKind.TEXT_ENCODE, TaskKind.VAE_DECODE, TaskKind.FINALIZE)
            if stage == STAGE_AUX
            else (TaskKind.DIT_PREPARE, TaskKind.TIMESTEP_PREPARE, TaskKind.DIT_STEP_CHUNK)
        )
        missing = [k.value for k in required_kinds if k not in group.supported_task_kinds]
        if missing:
            raise ValueError(
                f"runtime_v2 disaggregate {stage} group {group_id!r} is missing required "
                f"supported_task_kinds: {missing!r}"
            )

    @staticmethod
    def _all_task_kinds() -> tuple[TaskKind, ...]:
        return STANDARD_TASK_KINDS

    @classmethod
    def _build_topology(
        cls,
        omni_diffusion_config: Any,
        *,
        default_task_kinds: tuple[TaskKind, ...] | None = None,
    ) -> RuntimeTopology:
        parallel_config = omni_diffusion_config.parallel_config
        worker_count = int(omni_diffusion_config.num_gpus or parallel_config.world_size)
        workers = tuple(WorkerSpec(worker_rank=i, device_id=i) for i in range(worker_count))
        task_kinds = default_task_kinds or cls._all_task_kinds()
        groups_json = getattr(omni_diffusion_config, "runtime_v2_groups_json", None)
        group_sizes = getattr(omni_diffusion_config, "runtime_v2_group_sizes", None)
        legacy_dp_size = int(getattr(parallel_config, "data_parallel_size", 1) or 1)
        if legacy_dp_size > 1 and groups_json is None and group_sizes is None:
            raise ValueError(
                "runtime_v2 does not support legacy data_parallel_size > 1 in implicit single-group mode. "
                "Set data_parallel_size=1 and configure explicit execution groups via "
                "runtime_v2_group_sizes/runtime_v2_groups_json."
            )
        if groups_json is not None:
            groups = cls._build_groups_from_groups_json(
                groups_json=groups_json,
                worker_count=worker_count,
                default_task_kinds=task_kinds,
            )
        elif group_sizes is not None:
            groups = cls._build_groups_from_group_sizes(
                group_sizes=group_sizes,
                worker_count=worker_count,
                default_task_kinds=task_kinds,
            )
        elif (
            str(getattr(omni_diffusion_config, "runtime_v2_scheduler_policy", "") or "").lower()
            in {"edf_greedy", "edf_best_fit"}
        ):
            groups = cls._build_edf_greedy_bootstrap_groups(
                worker_count=worker_count,
                default_task_kinds=task_kinds,
            )
        else:
            default_ulysses_degree = int(
                getattr(parallel_config, "ulysses_degree", parallel_config.sequence_parallel_size or 1)
            )
            default_ring_degree = int(getattr(parallel_config, "ring_degree", 1))
            parallel_spec = ParallelSpec(
                tp=int(parallel_config.tensor_parallel_size),
                sp=int(parallel_config.sequence_parallel_size or 1),
                cfg=int(parallel_config.cfg_parallel_size),
            )
            groups = (
                ExecutionGroupSpec(
                    group_id="g0",
                    ranks=tuple(worker.worker_rank for worker in workers),
                    parallel_spec=parallel_spec,
                    supported_task_kinds=task_kinds,
                    ulysses_degree=default_ulysses_degree,
                    ring_degree=default_ring_degree,
                ),
            )

        cls._validate_group_partition(groups=groups, worker_count=worker_count)
        return RuntimeTopology(workers=workers, groups=groups)

    @classmethod
    def _build_edf_greedy_bootstrap_groups(
        cls,
        *,
        worker_count: int,
        default_task_kinds: tuple[TaskKind, ...],
    ) -> tuple[ExecutionGroupSpec, ...]:
        groups: list[ExecutionGroupSpec] = []
        dit_task_kinds = (TaskKind.DIT_PREPARE, TaskKind.TIMESTEP_PREPARE, TaskKind.DIT_STEP_CHUNK)
        if worker_count > 1:
            groups.append(
                ExecutionGroupSpec(
                    group_id="g_world",
                    ranks=tuple(range(worker_count)),
                    parallel_spec=ParallelSpec(tp=1, sp=worker_count, cfg=1),
                    supported_task_kinds=dit_task_kinds,
                    ulysses_degree=worker_count,
                    ring_degree=1,
                )
            )
        for rank in range(worker_count):
            groups.append(
                ExecutionGroupSpec(
                    group_id=f"g_sp1_r{rank}",
                    ranks=(rank,),
                    parallel_spec=ParallelSpec(tp=1, sp=1, cfg=1),
                    supported_task_kinds=default_task_kinds,
                    ulysses_degree=1,
                    ring_degree=1,
                )
            )
        return tuple(groups)

    @classmethod
    def _build_groups_from_group_sizes(
        cls,
        *,
        group_sizes: list[int] | tuple[int, ...] | str,
        worker_count: int,
        default_task_kinds: tuple[TaskKind, ...],
    ) -> tuple[ExecutionGroupSpec, ...]:
        sizes = cls._parse_group_sizes(group_sizes)
        if sum(sizes) != worker_count:
            raise ValueError(
                "runtime_v2_group_sizes must sum to available worker count "
                f"(sum={sum(sizes)}, workers={worker_count}). "
                "Check diffusion stage num_gpus/runtime.devices and CLI overrides."
            )
        groups: list[ExecutionGroupSpec] = []
        cursor = 0
        for idx, size in enumerate(sizes):
            ranks = tuple(range(cursor, cursor + size))
            cursor += size
            groups.append(
                ExecutionGroupSpec(
                    group_id=f"g{idx}",
                    ranks=ranks,
                    parallel_spec=ParallelSpec(tp=size, sp=1, cfg=1),
                    supported_task_kinds=default_task_kinds,
                    ulysses_degree=1,
                    ring_degree=1,
                )
            )
        return tuple(groups)

    @classmethod
    def _build_groups_from_groups_json(
        cls,
        *,
        groups_json: list[dict[str, Any]],
        worker_count: int,
        default_task_kinds: tuple[TaskKind, ...],
    ) -> tuple[ExecutionGroupSpec, ...]:
        if not isinstance(groups_json, list):
            raise ValueError("runtime_v2_groups_json must be a list of group objects")
        if not groups_json:
            raise ValueError("runtime_v2_groups_json must contain at least one group")
        groups: list[ExecutionGroupSpec] = []
        next_rank = 0
        for idx, raw_group in enumerate(groups_json):
            if not isinstance(raw_group, Mapping):
                raise ValueError(
                    f"runtime_v2_groups_json entry #{idx} must be an object, got {type(raw_group)!r}"
                )
            group_id = str(raw_group.get("group_id", f"g{idx}"))
            ranks_raw = raw_group.get("ranks")
            size_raw = raw_group.get("size")
            if ranks_raw is not None:
                if not isinstance(ranks_raw, (list, tuple)):
                    raise ValueError(f"group {group_id!r} field 'ranks' must be a list of ints")
                ranks = tuple(int(rank) for rank in ranks_raw)
                if not ranks:
                    raise ValueError(f"group {group_id!r} must contain at least one rank")
                group_size = len(ranks)
                if size_raw is not None and int(size_raw) != group_size:
                    raise ValueError(
                        f"group {group_id!r} size mismatch: size={int(size_raw)} but len(ranks)={group_size}"
                    )
            else:
                if size_raw is None:
                    raise ValueError(
                        f"group {group_id!r} must provide either 'ranks' or 'size' in runtime_v2_groups_json"
                    )
                group_size = int(size_raw)
                if group_size < 1:
                    raise ValueError(f"group {group_id!r} size must be >= 1, got {group_size}")
                ranks = tuple(range(next_rank, next_rank + group_size))
                next_rank += group_size

            tp = int(raw_group.get("tp", group_size))
            sp, ulysses_degree, ring_degree = cls._resolve_sequence_parallel_dims(
                raw_group=raw_group,
                group_id=group_id,
            )
            cfg = int(raw_group.get("cfg", 1))
            cfg_parallel = bool(raw_group.get("cfg_parallel", False))
            parallel_spec = ParallelSpec(tp=tp, sp=sp, cfg=cfg, cfg_parallel=cfg_parallel)
            expected_world_size = int(parallel_spec.tp) * int(parallel_spec.sp) * int(parallel_spec.cfg)
            if expected_world_size != len(ranks):
                raise ValueError(
                    f"group {group_id!r} parallel mismatch: tp*sp*cfg={expected_world_size} "
                    f"!= group size {len(ranks)}"
                )

            task_kinds = cls._parse_group_task_kinds(raw_group.get("supported_task_kinds"), default_task_kinds)
            groups.append(
                ExecutionGroupSpec(
                    group_id=group_id,
                    ranks=ranks,
                    parallel_spec=parallel_spec,
                    supported_task_kinds=task_kinds,
                    ulysses_degree=ulysses_degree,
                    ring_degree=ring_degree,
                )
            )
        return tuple(groups)

    @staticmethod
    def _parse_group_sizes(value: list[int] | tuple[int, ...] | str) -> tuple[int, ...]:
        if isinstance(value, str):
            parts = [part.strip() for part in value.split(",") if part.strip()]
            if not parts:
                raise ValueError("runtime_v2_group_sizes string cannot be empty")
            sizes = tuple(int(part) for part in parts)
        elif isinstance(value, (list, tuple)):
            sizes = tuple(int(part) for part in value)
        else:
            raise ValueError(f"runtime_v2_group_sizes must be list[int] or comma string, got {type(value)!r}")
        if not sizes:
            raise ValueError("runtime_v2_group_sizes must not be empty")
        if any(size < 1 for size in sizes):
            raise ValueError(f"runtime_v2_group_sizes must be positive integers, got {list(sizes)!r}")
        return sizes

    @staticmethod
    def _parse_group_task_kinds(
        raw_task_kinds: Any,
        default_task_kinds: tuple[TaskKind, ...],
    ) -> tuple[TaskKind, ...]:
        if raw_task_kinds is None:
            return default_task_kinds
        if not isinstance(raw_task_kinds, (list, tuple)):
            raise ValueError("supported_task_kinds must be a list of task kind strings")
        parsed = tuple(TaskKind(str(kind)) for kind in raw_task_kinds)
        if not parsed:
            raise ValueError("supported_task_kinds must not be empty")
        return parsed

    @staticmethod
    def _resolve_sequence_parallel_dims(
        *,
        raw_group: Mapping[str, Any],
        group_id: str,
    ) -> tuple[int, int, int]:
        legacy_sp = raw_group.get("sp", raw_group.get("sequence_parallel_size"))
        ulysses_raw = raw_group.get("ulysses_degree", raw_group.get("ulysses"))
        ring_raw = raw_group.get("ring_degree", raw_group.get("ring"))

        has_ulysses = ulysses_raw is not None
        has_ring = ring_raw is not None
        has_legacy_sp = legacy_sp is not None

        if not has_ulysses and not has_ring:
            if has_legacy_sp:
                sp = int(legacy_sp)
                ulysses_degree = sp
                ring_degree = 1
            else:
                ulysses_degree = 1
                ring_degree = 1
                sp = 1
        else:
            ulysses_degree = int(ulysses_raw if has_ulysses else 1)
            ring_degree = int(ring_raw if has_ring else 1)
            sp = ulysses_degree * ring_degree
            if has_legacy_sp and int(legacy_sp) != sp:
                raise ValueError(
                    f"group {group_id!r} sequence parallel mismatch: "
                    f"sp={int(legacy_sp)} != ulysses_degree({ulysses_degree}) * ring_degree({ring_degree})"
                )

        if ulysses_degree < 1:
            raise ValueError(f"group {group_id!r} ulysses_degree must be >= 1, got {ulysses_degree}")
        if ring_degree < 1:
            raise ValueError(f"group {group_id!r} ring_degree must be >= 1, got {ring_degree}")
        if sp < 1:
            raise ValueError(f"group {group_id!r} sequence parallel size must be >= 1, got {sp}")
        return sp, ulysses_degree, ring_degree

    @staticmethod
    def _validate_group_partition(
        *,
        groups: tuple[ExecutionGroupSpec, ...],
        worker_count: int,
    ) -> None:
        all_worker_ranks = set(range(worker_count))
        assigned: set[int] = set()
        out_of_range: set[int] = set()
        for group in groups:
            for rank in group.ranks:
                if rank < 0 or rank >= worker_count:
                    out_of_range.add(rank)
                assigned.add(rank)
        if out_of_range:
            raise ValueError(
                f"runtime_v2 groups contain out-of-range worker ranks: {sorted(out_of_range)!r}, "
                f"worker_count={worker_count}"
            )
        missing = sorted(all_worker_ranks - assigned)
        if missing:
            raise ValueError(f"runtime_v2 groups do not cover all workers, missing ranks: {missing!r}")

    def _to_runtime_request(self, request: Any, *, denoise_chunk_size: int | None = None) -> Any:
        return self.adapter.normalize_request(
            request,
            denoise_chunk_size=int(denoise_chunk_size or self.default_step_chunk_size),
        )

    def submit(self, request: Any, *, denoise_chunk_size: int | None = None) -> str:
        runtime_req = self._to_runtime_request(request, denoise_chunk_size=denoise_chunk_size)
        req = runtime_req.diffusion_request
        sp = req.sampling_params
        logger.info(
            "runtime_v2 submit: request_id=%s model_class_name=%s chunk=%s steps=%s frames=%s size=%sx%s",
            runtime_req.request_id,
            getattr(self.od_config, "model_class_name", None),
            runtime_req.denoise_chunk_size,
            getattr(sp, "num_inference_steps", None),
            getattr(sp, "num_frames", None),
            getattr(sp, "width", None),
            getattr(sp, "height", None),
        )
        return self.scheduler.submit_request(runtime_req)

    def poll_once(self, timeout_s: float = 0.0) -> None:
        self.scheduler.poll_once(timeout_s=timeout_s)

    def get_request_status(self, request_id: str) -> tuple[str, Any | None]:
        return self.scheduler.get_request_status(request_id)

    def _release_request(self, request_id: str) -> None:
        # Retire all controller-side state for a request once its terminal status
        # has been delivered, so plans / artifact store / completed-output cache do
        # not accumulate for the process lifetime.
        scheduler = getattr(self, "scheduler", None)
        if scheduler is not None and hasattr(scheduler, "release_request"):
            scheduler.release_request(request_id)

    def release_request(self, request_id: str) -> None:
        self._release_request(request_id)

    def wait(self, request_id: str, *, timeout_s: float | None = None):
        if timeout_s is not None and timeout_s <= 0:
            status, payload = self.get_request_status(request_id)
            if status == "failed":
                logger.error("runtime_v2 wait failed: request_id=%s elapsed=%.3fs", request_id, 0.0)
                self._release_request(request_id)
                raise RuntimeError(payload)
            if status == "finished":
                logger.info("runtime_v2 wait finished: request_id=%s elapsed=%.3fs", request_id, 0.0)
                self._release_request(request_id)
                return payload
            self.poll_once(timeout_s=0.0)
            status, payload = self.get_request_status(request_id)
            if status == "failed":
                logger.error("runtime_v2 wait failed: request_id=%s elapsed=%.3fs", request_id, 0.0)
                self._release_request(request_id)
                raise RuntimeError(payload)
            if status == "finished":
                logger.info("runtime_v2 wait finished: request_id=%s elapsed=%.3fs", request_id, 0.0)
                self._release_request(request_id)
                return payload
            raise TimeoutError(f"request {request_id} timed out")

        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        start_t = time.monotonic()
        last_progress_log_t = 0.0
        while True:
            status, payload = self.get_request_status(request_id)
            if status == "failed":
                logger.error(
                    "runtime_v2 wait failed: request_id=%s elapsed=%.3fs",
                    request_id,
                    time.monotonic() - start_t,
                )
                self._release_request(request_id)
                raise RuntimeError(payload)
            if status == "finished":
                logger.info(
                    "runtime_v2 wait finished: request_id=%s elapsed=%.3fs",
                    request_id,
                    time.monotonic() - start_t,
                )
                self._release_request(request_id)
                return payload
            now = time.monotonic()
            if now - last_progress_log_t >= 2.0:
                scheduler = getattr(self, "scheduler", None)
                if scheduler is not None:
                    progress = scheduler.get_request_progress(request_id)
                    logger.info(
                        "runtime_v2 wait progress: request_id=%s finished=%s/%s running=%s queued=%s",
                        request_id,
                        progress["finished_tasks"],
                        progress["total_tasks"],
                        progress["running_tasks"],
                        progress["pending_tasks"],
                    )
                last_progress_log_t = now
            sleep_s = self._poll_interval_s
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    logger.error(
                        "runtime_v2 wait timeout: request_id=%s elapsed=%.3fs",
                        request_id,
                        time.monotonic() - start_t,
                    )
                    raise TimeoutError(f"request {request_id} timed out")
                sleep_s = min(sleep_s, remaining)
            try:
                self.poll_once(timeout_s=sleep_s)
            except Exception as exc:
                raise RuntimeError(f"runtime_v2 poll failed while waiting for {request_id}: {exc}") from exc

    def execute(self, request: Any, *, denoise_chunk_size: int | None = None, timeout_s: float | None = None):
        request_id = self.submit(request, denoise_chunk_size=denoise_chunk_size)
        return self.wait(request_id, timeout_s=timeout_s)

    def close(self) -> None:
        self.scheduler.shutdown()
        logger.info("runtime_v2 runner closed")

    def collective_rpc(
        self,
        method: str,
        *,
        timeout_s: float | None = None,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
        unique_reply_rank: int | None = None,
    ) -> list[Any] | Any:
        raise NotImplementedError(
            "runtime_v2 collective_rpc is temporarily disabled while focusing on async artifact fetch path"
        )

    def check_health(self) -> None:
        if hasattr(self.worker_pool, "check_health"):
            self.worker_pool.check_health()
