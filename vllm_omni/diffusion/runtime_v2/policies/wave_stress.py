# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from typing import Any

from vllm.logger import init_logger

from vllm_omni.diffusion.runtime_v2.policies.disaggregate import (
    DynamicStepFCFSSchedulerPolicy,
)
from vllm_omni.diffusion.runtime_v2.protocol import (
    ExecutionGroupSpec,
    InferenceTask,
    ParallelSpec,
    RequestExecutionPlan,
    TaskKind,
    TaskStatus,
    WorkerEvent,
    WorkerEventKind,
)
from vllm_omni.diffusion.runtime_v2.topology import RuntimeTopology

logger = init_logger(__name__)


_WAVE_ROLES: tuple[str, ...] = ("cross_g01", "cross_g12", "cross_full", "pingpong")
_PINGPONG_CYCLE: tuple[str, ...] = ("stress_gfull", "stress_g01", "stress_g12", "stress_gfull")
_STRESS_TASK_KINDS: tuple[TaskKind, ...] = (
    TaskKind.TEXT_ENCODE,
    TaskKind.DIT_PREPARE,
    TaskKind.TIMESTEP_PREPARE,
    TaskKind.DIT_STEP_CHUNK,
    TaskKind.VAE_DECODE,
    TaskKind.FINALIZE,
)


class WaveStressSchedulerPolicy(DynamicStepFCFSSchedulerPolicy):
    """Deterministic stress policy for runtime_v2.

    First ``warmup_reqs`` requests run through the parent DynamicStepFCFS
    policy unchanged. Subsequent requests are buffered until a full wave
    of ``wave_size`` is collected; the wave is then activated with internal
    roles assigned by arrival order:

      wave[0] -> "cross_g01"   (DIT chunks pinned to G01,  ranks (0, 1))
      wave[1] -> "cross_g12"   (DIT chunks pinned to G12,  ranks (1, 2))
      wave[2] -> "cross_full"  (DIT chunks pinned to GFULL,ranks (0,1,2,3))
      wave[3] -> "pingpong"    (DIT chunks rotate GFULL -> G01 -> G12 -> GFULL)

    Wave DIT chunks are dispatched in strict round-robin across the four
    wave requests, forcing the worker that is shared by multiple groups
    (rank 1, in the (0,1)/(1,2)/(0,1,2,3) layout) to switch its SP session
    on every consecutive task. The pingpong request additionally exercises
    same-request reshard via the scheduler's implicit
    `_migrate_worker_local_input_for_task` path.

    The policy registers three GFC subgroups (stress_g01 / stress_g12 /
    stress_gfull) into the topology at construction time. The dynamic-group
    machinery in :class:`RuntimeTopology` and the worker's
    `_ensure_group_spec` path materializes their `_FixedParallelSession` on
    first dispatch. This policy only makes sense under the GFC collective
    backend and refuses to construct otherwise.

    Request-count caveat — READ BEFORE USING
    ----------------------------------------
    This is a **stress test policy**, not a production scheduler. It only
    releases a wave when ``wave_size`` post-warmup requests have arrived.
    A trailing partial pool (1..wave_size-1) auto-flushes when the *previous*
    active wave drains, but if a wave never forms in the first place (e.g.
    caller submits ``warmup_reqs + N`` requests with 0 < N < wave_size and
    then goes idle), **the trailing N requests hang forever** — there is no
    background timer and no serving-side flush hook.

    The intended usage pattern is a closed-loop benchmark that submits
    ``warmup_reqs + k * wave_size`` requests in a single burst, waits for
    all to complete, then exits. Do not use this policy for interactive
    serving, A/B tests with arbitrary request counts, or any deployment
    where the operator cannot guarantee the wave-aligned total.
    """

    def __init__(
        self,
        topology: RuntimeTopology,
        *,
        warmup_reqs: int = 5,
        wave_size: int = 4,
        aux_group_id: str | None = None,
        collective_backend: str | None = None,
        require_gfc_backend: bool = True,
        stage_group_ids: dict[str, str] | None = None,
        dit_step_schedule: tuple[dict[str, Any], ...] | None = None,
    ) -> None:
        super().__init__(
            topology=topology,
            stage_group_ids=stage_group_ids,
            dit_step_schedule=dit_step_schedule,
        )
        if require_gfc_backend and collective_backend is not None and collective_backend.lower() != "gfc":
            # Backend was passed explicitly (production path: runner reads it from
            # od_config). Reject torch directly. When collective_backend is None
            # (programmatic construction without explicit backend), trust the
            # caller — data.py / runner.py already validate the GFC requirement
            # via the OmniDiffusionConfig __post_init__ path.
            raise RuntimeError(
                "WaveStressSchedulerPolicy requires runtime_v2_collective_backend='gfc'; "
                f"got {collective_backend!r}. The dynamic G01/G12/GFULL subgroups it "
                "registers cannot be materialized under the torch backend (no startup-"
                "time NCCL handshake for them)."
            )
        if wave_size < 2 or wave_size > len(_WAVE_ROLES):
            raise ValueError(
                f"wave_size must be in [2, {len(_WAVE_ROLES)}], got {wave_size}"
            )
        if len(self.topology.workers) < 4:
            raise RuntimeError(
                "WaveStressSchedulerPolicy needs at least 4 workers to register "
                f"G01/G12/GFULL subgroups, topology has {len(self.topology.workers)}"
            )
        self._warmup_reqs = int(warmup_reqs)
        self._wave_size = int(wave_size)
        self._num_seen_reqs = 0
        # Plans buffered until a full wave is collected. Each entry is the
        # plan; root tasks are re-derived at activation time.
        self._held_plans: list[RequestExecutionPlan] = []
        self._active_wave_order: list[str] = []
        self._req_role: dict[str, str] = {}
        # Per-pingpong-request DIT chunk index, used to walk _PINGPONG_CYCLE.
        self._pingpong_chunk_idx: dict[str, int] = {}
        self._rr_cursor: int = 0
        # Per-wave-request ready queue. Wave-request tasks bypass the parent's
        # `group_ready[group_id]` queue so we can dispatch them in round-robin
        # request order instead of stable topology-group order.
        self._wave_ready_by_req: dict[str, deque[InferenceTask]] = {}
        self._ensure_stress_subgroups()
        self._aux_group_id = aux_group_id or "stress_gfull"
        # Sanity-check the aux group exists in topology after subgroup setup.
        self.topology.get_group(self._aux_group_id)

    # ---- subgroup registration ----

    def _ensure_stress_subgroups(self) -> None:
        for group_id, ranks, sp in (
            ("stress_g01", (0, 1), 2),
            ("stress_g12", (1, 2), 2),
            ("stress_gfull", (0, 1, 2, 3), 4),
        ):
            self.topology.ensure_group(
                ExecutionGroupSpec(
                    group_id=group_id,
                    ranks=ranks,
                    parallel_spec=ParallelSpec(tp=1, sp=sp, cfg=1),
                    supported_task_kinds=_STRESS_TASK_KINDS,
                    ulysses_degree=sp,
                    ring_degree=1,
                )
            )

    # ---- admission ----

    def on_request_submitted(self, plan: RequestExecutionPlan) -> Iterable[InferenceTask]:
        self._num_seen_reqs += 1
        if self._num_seen_reqs <= self._warmup_reqs:
            return super().on_request_submitted(plan)
        self._held_plans.append(plan)
        if not self._active_wave_order and len(self._held_plans) >= self._wave_size:
            self._activate_next_wave()
            return self._take_dispatchable_tasks()
        return []

    def _activate_next_wave(self) -> None:
        assert not self._active_wave_order, "previous wave must drain before activating next"
        wave_plans = self._held_plans[: self._wave_size]
        self._held_plans = self._held_plans[self._wave_size :]
        roles = _WAVE_ROLES[: self._wave_size]
        for plan, role in zip(wave_plans, roles, strict=True):
            self._req_role[plan.request_id] = role
            if role == "pingpong":
                self._pingpong_chunk_idx[plan.request_id] = 0
            self._active_wave_order.append(plan.request_id)
        self._rr_cursor = 0
        logger.info(
            "wave_stress activated wave: %s",
            [(rid, self._req_role[rid]) for rid in self._active_wave_order],
        )
        # Enqueue root tasks for every wave request through the normal pipeline,
        # but suppress immediate dispatch — caller will invoke _take_dispatchable_tasks.
        for plan in wave_plans:
            root_tasks = [task for task in plan.tasks.values() if not task.dependencies]
            self._enqueue_runnable(root_tasks)

    # ---- runnable bookkeeping ----

    def on_tasks_runnable(self, tasks: Iterable[InferenceTask]) -> Iterable[InferenceTask]:
        self._enqueue_runnable(tasks)
        return self._take_dispatchable_tasks()

    def _enqueue_runnable(self, tasks: Iterable[InferenceTask]) -> None:
        for task in tasks:
            if task.group_id is None:
                task.group_id = self._select_group_for_task(task)
            self._validate_task(task)
            task.status = TaskStatus.READY
            if task.kind == TaskKind.RESHARD:
                self.reshard_ready.append(task)
            elif task.request_id in self._req_role:
                self._wave_ready_by_req.setdefault(task.request_id, deque()).append(task)
            else:
                assert task.group_id is not None
                self.group_ready.setdefault(task.group_id, deque()).append(task)

    # ---- group selection ----

    def _select_group_for_task(self, task: InferenceTask) -> str | None:
        role = self._req_role.get(task.request_id)
        if role is None:
            # Warmup / non-wave request. Prefer parent's stage_group_ids /
            # dit_step_schedule if the operator wired them, otherwise fall back
            # to the policy's own self-registered stress_gfull so no static
            # disaggregate config is needed.
            from_super = super()._select_group_for_task(task)
            return from_super if from_super is not None else self._aux_group_id
        if task.kind != TaskKind.DIT_STEP_CHUNK:
            return self._aux_group_id
        if role == "cross_g01":
            return "stress_g01"
        if role == "cross_g12":
            return "stress_g12"
        if role == "cross_full":
            return "stress_gfull"
        if role == "pingpong":
            idx = self._pingpong_chunk_idx[task.request_id]
            self._pingpong_chunk_idx[task.request_id] = idx + 1
            return _PINGPONG_CYCLE[idx % len(_PINGPONG_CYCLE)]
        raise ValueError(f"wave_stress: unknown role {role!r} for task {task.task_id}")

    # ---- dispatch ----

    def _take_dispatchable_tasks(self) -> list[InferenceTask]:
        out = list(super()._take_dispatchable_tasks())
        out.extend(self._take_wave_dispatchable())
        return out

    def _take_wave_dispatchable(self) -> list[InferenceTask]:
        out: list[InferenceTask] = []
        if not self._active_wave_order:
            return out
        n = len(self._active_wave_order)
        attempts = 0
        # Each iteration tries to dispatch the head task of the request at the
        # current cursor. Successful dispatch advances cursor + resets attempts;
        # unsuccessful advances cursor + increments attempts. We stop after a
        # full lap with no dispatch (no wave request can make progress now).
        while attempts < n:
            req_id = self._active_wave_order[self._rr_cursor]
            queue = self._wave_ready_by_req.get(req_id)
            dispatched = False
            if queue:
                task = queue[0]
                group = self.topology.get_group(task.group_id or "")
                ranks = tuple(int(rank) for rank in group.ranks)
                if self._can_dispatch_ranks(ranks):
                    queue.popleft()
                    if not queue:
                        self._wave_ready_by_req.pop(req_id, None)
                    task.status = TaskStatus.DISPATCHED
                    self._claim_ranks(ranks, reshard=False)
                    self.outstanding_per_group[task.group_id] = (
                        self.outstanding_per_group.get(task.group_id, 0) + 1
                    )
                    self.dispatched_kind[task.task_id] = task.group_id
                    self.dispatched_ranks_by_task[task.task_id] = ranks
                    out.append(task)
                    dispatched = True
                    logger.info(
                        "wave_stress dispatch: cursor=%s req=%s role=%s kind=%s group=%s ranks=%s task_id=%s",
                        self._rr_cursor,
                        req_id,
                        self._req_role.get(req_id),
                        task.kind.value,
                        task.group_id,
                        ranks,
                        task.task_id,
                    )
            self._rr_cursor = (self._rr_cursor + 1) % n
            if dispatched:
                attempts = 0
            else:
                attempts += 1
        return out

    # ---- lifecycle: release / drop ----

    def on_worker_event(self, event: WorkerEvent) -> Iterable[InferenceTask]:
        if event.request_id in self._req_role and event.kind in (
            WorkerEventKind.TASK_EXEC_END,
            WorkerEventKind.TASK_FAILED,
            WorkerEventKind.REQUEST_FINISHED,
            WorkerEventKind.REQUEST_FAILED,
        ):
            logger.info(
                "wave_stress event: kind=%s request_id=%s role=%s task_id=%s group=%s",
                event.kind.value,
                event.request_id,
                self._req_role.get(event.request_id),
                event.task_id or "-",
                event.group_id or "-",
            )
        return super().on_worker_event(event)

    def _drop_request(self, request_id: str) -> None:
        super()._drop_request(request_id)
        self._wave_ready_by_req.pop(request_id, None)
        if request_id in self._req_role:
            del self._req_role[request_id]
            self._pingpong_chunk_idx.pop(request_id, None)
            if request_id in self._active_wave_order:
                self._active_wave_order.remove(request_id)
                if self._active_wave_order:
                    self._rr_cursor %= len(self._active_wave_order)
                else:
                    self._rr_cursor = 0
        if not self._active_wave_order:
            if len(self._held_plans) >= self._wave_size:
                self._activate_next_wave()
            elif self._held_plans:
                # No wave can form right now and the active wave just drained.
                # Release trailing held plans through the warmup path so they
                # don't hang waiting for never-arriving siblings. This is the
                # eager-flush policy: stress runs typically submit all prompts
                # up front, and a tail of (1..wave_size-1) plans would otherwise
                # block forever. Real interactive serving should not use
                # wave_stress.
                self._flush_held_plans_as_warmup()

    def _flush_held_plans_as_warmup(self) -> None:
        if not self._held_plans:
            return
        plans = self._held_plans
        self._held_plans = []
        logger.info(
            "wave_stress flushing %d trailing held plan(s) through warmup path: %s",
            len(plans),
            [plan.request_id for plan in plans],
        )
        for plan in plans:
            root_tasks = [task for task in plan.tasks.values() if not task.dependencies]
            self._enqueue_runnable(root_tasks)
