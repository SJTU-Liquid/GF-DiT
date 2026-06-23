# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import time
from collections import deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from vllm.logger import init_logger

from vllm_omni.diffusion.runtime_v2.policies.disaggregate import (
    DynamicStepFCFSSchedulerPolicy,
)
from vllm_omni.diffusion.runtime_v2.policies.common import (
    TaskRuntimeEstimator,
    _resolve_request_stage,
    _resolve_seq_len,
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


@dataclass(frozen=True)
class _EdfGreedyRequestMeta:
    deadline_ms: float
    priority: int
    arrival_ms: float
    order: int


class EdfGreedySchedulerPolicy(DynamicStepFCFSSchedulerPolicy):
    """EDF request ordering with greedy free-rank SP placement.

    This is the production runtime variant of the simulator's EDF+greedy
    policy. It assigns a group when a task becomes dispatchable instead of at
    request admission. Single-rank stages use one-rank groups so VAE/text work
    does not occupy an entire DiT SP group. DiT stages take the largest allowed
    SP size that fits in the current free-rank pool, synthesizing a runtime_v2
    execution group on demand when the exact rank set is not already present.
    """

    _SINGLE_RANK_TASK_KINDS = (TaskKind.TEXT_ENCODE, TaskKind.VAE_DECODE, TaskKind.FINALIZE)
    _DIT_TASK_KINDS = (TaskKind.DIT_PREPARE, TaskKind.TIMESTEP_PREPARE, TaskKind.DIT_STEP_CHUNK)
    _STANDARD_TASK_KINDS = _SINGLE_RANK_TASK_KINDS + _DIT_TASK_KINDS

    def __init__(
        self,
        topology: RuntimeTopology,
        task_runtime_estimator: TaskRuntimeEstimator | None = None,
        *,
        allowed_sp_sizes: Iterable[int] | None = None,
    ) -> None:
        super().__init__(topology=topology)
        self.task_runtime_estimator = task_runtime_estimator
        self.ready_by_request: dict[str, deque[InferenceTask]] = {}
        self.request_meta: dict[str, _EdfGreedyRequestMeta] = {}
        # Cache plans so _select_dit_group can resolve seq_len for the
        # cost-aware SP picker without re-walking the scheduler.
        self.plans: dict[str, RequestExecutionPlan] = {}
        self.request_inflight: dict[str, int] = {}
        self.request_last_group: dict[str, str] = {}
        self._task_index: dict[str, InferenceTask] = {}
        self._request_seq = 0
        self.now_ms = 0.0
        self.allowed_sp_sizes = self._normalize_allowed_sp_sizes(allowed_sp_sizes)

    def on_request_submitted(self, plan: RequestExecutionPlan) -> Iterable[InferenceTask]:
        self._request_seq += 1
        self.request_meta[plan.request_id] = self._meta_from_plan(plan, self._request_seq)
        self.plans[plan.request_id] = plan
        for task in plan.tasks.values():
            self._task_index[task.task_id] = task
        root_tasks = [task for task in plan.tasks.values() if not task.dependencies]
        return self.on_tasks_runnable(root_tasks)

    def on_tasks_runnable(self, tasks: Iterable[InferenceTask]) -> Iterable[InferenceTask]:
        for task in tasks:
            task.status = TaskStatus.READY
            self._task_index[task.task_id] = task
            if task.group_id is not None:
                self._validate_explicit_task_group(task)
            self.ready_by_request.setdefault(task.request_id, deque()).append(task)
        return self._take_dispatchable_tasks()

    def on_worker_event(self, event: WorkerEvent) -> Iterable[InferenceTask]:
        self.now_ms = max(self.now_ms, float(event.timestamp_ns) / 1_000_000.0)
        if event.kind in (WorkerEventKind.TASK_EXEC_END, WorkerEventKind.TASK_FAILED):
            self._release_dispatched_task(event)
        if event.kind in (WorkerEventKind.REQUEST_FINISHED, WorkerEventKind.REQUEST_FAILED):
            self._drop_request(event.request_id)
        return self._take_dispatchable_tasks()

    def _take_dispatchable_tasks(self) -> list[InferenceTask]:
        free_ranks = self._free_rank_set()
        ready: list[tuple[str, InferenceTask]] = []
        for request_id, queue in self.ready_by_request.items():
            if not queue:
                continue
            if self.request_inflight.get(request_id, 0) > 0:
                continue
            ready.append((request_id, queue[0]))

        ready.sort(key=lambda item: self._request_sort_key(item[0]))
        out: list[InferenceTask] = []
        for request_id, task in ready:
            placement = self._select_placement(task, free_ranks)
            if placement is None:
                continue
            group_id, ranks = placement
            self._dispatch_task(task, request_id, group_id, ranks)
            free_ranks.difference_update(ranks)
            out.append(task)
        return out

    def _select_placement(
        self,
        task: InferenceTask,
        free_ranks: set[int],
    ) -> tuple[str, tuple[int, ...]] | None:
        if task.group_id is not None:
            group = self.topology.get_group(task.group_id)
            ranks = tuple(int(rank) for rank in group.ranks)
            if task.kind in self._SINGLE_RANK_TASK_KINDS and len(ranks) != 1:
                raise ValueError(
                    f"edf_greedy requires single-rank group for {task.kind.value}; "
                    f"task {task.task_id} is pinned to {task.group_id!r} with ranks={ranks!r}"
                )
            if set(ranks).issubset(free_ranks):
                return group.group_id, ranks
            return None

        if task.kind in self._SINGLE_RANK_TASK_KINDS:
            return self._select_single_rank_group(task, free_ranks)
        if task.kind in self._DIT_TASK_KINDS:
            return self._select_dit_group(task, free_ranks)
        raise ValueError(f"edf_greedy does not support task kind {task.kind.value!r}")

    def _select_single_rank_group(
        self,
        task: InferenceTask,
        free_ranks: set[int],
    ) -> tuple[str, tuple[int, ...]] | None:
        for rank in sorted(free_ranks):
            group = self._ensure_group_for_ranks(
                ranks=(rank,),
                supported_task_kinds=self._STANDARD_TASK_KINDS,
            )
            if task.kind not in group.supported_task_kinds:
                continue
            return group.group_id, (rank,)
        return None

    def _pick_ranks_for_request(
        self,
        request_id: str,
        sp_size: int,
        free_ranks: set[int],
    ) -> tuple[int, ...]:
        # Pick `sp_size` ranks for this request, preferring the same set the
        # request's previous task ran on when it is still entirely free at the
        # same width. Without this, a request that just finished a DiT chunk on
        # rank 2 gets re-dispatched to rank 0 the next tick (because the free
        # set's lowest is 0) -- forcing the artifact to migrate 2->0 even
        # though rank 2 stayed free. Under heavy load (edf_best_fit case) that
        # gratuitous churn dominates the migration count: measured 2026-05-23,
        # best_fit without stickiness ran 227 migrations vs greedy's 201 in
        # the same 4-GPU run, costing ~6 extra S-class SLO misses.
        last_group_id = self.request_last_group.get(request_id)
        if last_group_id is not None:
            try:
                last_ranks = tuple(self.topology.get_group(last_group_id).ranks)
            except (KeyError, AttributeError):
                last_ranks = ()
            if (
                len(last_ranks) == sp_size
                and set(last_ranks).issubset(free_ranks)
            ):
                return last_ranks
        return tuple(sorted(free_ranks)[:sp_size])

    def _select_dit_group(
        self,
        task: InferenceTask,
        free_ranks: set[int],
    ) -> tuple[str, tuple[int, ...]] | None:
        if not free_ranks:
            return None
        # Candidates: SP sizes that fit in free_ranks.
        fitting = [s for s in self.allowed_sp_sizes if s <= len(free_ranks)]
        if not fitting:
            return None
        # Default order: largest first (original greedy behavior).
        candidates_order = sorted(fitting, reverse=True)
        # If we have a cost-model-backed runtime estimator, reorder
        # candidates by estimated per-task cost (smallest first), tie-broken
        # by smaller sp (fewer ranks used). This prevents picking a wider SP
        # whose collective overhead makes it actually SLOWER than a smaller
        # SP for THIS request -- e.g., Qwen Image S@512x512:
        #   SP1: 155 ms/step, SP2: 161, SP4: 192 (GFC comm dominates).
        # Without this, free 4 ranks + S task --> pick SP4 --> 25 % slower
        # per step AND blocks all other parallel work.
        plan = self.plans.get(task.request_id)
        if self.task_runtime_estimator is not None and plan is not None:
            seq_len = _resolve_seq_len(task, plan)
            stage = _resolve_request_stage(task)

            def _cost(sp: int) -> tuple[float, int]:
                spec = ParallelSpec(tp=1, sp=sp, cfg=1)
                c = self.task_runtime_estimator(spec, seq_len, stage)
                # Estimator returns 0.0 when no cost-model entry exists
                # (see _make_cost_model_estimator in runner.py). Sort that
                # SP LAST so we don't prefer a wider SP that's "unknown".
                return (c if c > 0 else float("inf"), sp)

            candidates_order = sorted(fitting, key=_cost)
        for sp_size in candidates_order:
            ranks = self._pick_ranks_for_request(
                task.request_id, sp_size, free_ranks
            )
            group = self._ensure_group_for_ranks(
                ranks=ranks,
                supported_task_kinds=self._DIT_TASK_KINDS,
            )
            if task.kind in group.supported_task_kinds:
                return group.group_id, ranks
        return None

    def _dispatch_task(
        self,
        task: InferenceTask,
        request_id: str,
        group_id: str,
        ranks: tuple[int, ...],
    ) -> None:
        queue = self.ready_by_request[request_id]
        if not queue or queue[0].task_id != task.task_id:
            return
        queue.popleft()
        if not queue:
            self.ready_by_request.pop(request_id, None)
        task.group_id = group_id
        task.status = TaskStatus.DISPATCHED
        self._claim_ranks(ranks, reshard=False)
        self.outstanding_per_group[group_id] = self.outstanding_per_group.get(group_id, 0) + 1
        self.dispatched_kind[task.task_id] = group_id
        self.dispatched_ranks_by_task[task.task_id] = ranks
        self.request_inflight[request_id] = self.request_inflight.get(request_id, 0) + 1

    def _release_dispatched_task(self, event: WorkerEvent) -> None:
        target = self.dispatched_kind.pop(event.task_id, None)
        ranks = self.dispatched_ranks_by_task.pop(event.task_id, ())
        if target is not None:
            remaining = self.outstanding_per_group.get(target, 0)
            if remaining <= 1:
                self.outstanding_per_group.pop(target, None)
            else:
                self.outstanding_per_group[target] = remaining - 1
            self._release_rank_counts(self.outstanding_per_rank, ranks)
        request_id = event.request_id
        inflight = self.request_inflight.get(request_id, 0)
        if inflight <= 1:
            self.request_inflight.pop(request_id, None)
        else:
            self.request_inflight[request_id] = inflight - 1
        if event.group_id:
            self.request_last_group[request_id] = event.group_id

    def _drop_request(self, request_id: str) -> None:
        self.ready_by_request.pop(request_id, None)
        self.request_meta.pop(request_id, None)
        self.plans.pop(request_id, None)
        self.request_inflight.pop(request_id, None)
        self.request_last_group.pop(request_id, None)

    def _free_rank_set(self) -> set[int]:
        free: set[int] = set()
        for worker in self.topology.workers:
            rank = int(worker.worker_rank)
            if self.outstanding_per_rank.get(rank, 0) > 0:
                continue
            if self.reshard_holds_by_rank.get(rank, 0) > 0:
                continue
            free.add(rank)
        return free

    def _ensure_group_for_ranks(
        self,
        *,
        ranks: tuple[int, ...],
        supported_task_kinds: tuple[TaskKind, ...],
    ) -> ExecutionGroupSpec:
        ranks = tuple(int(rank) for rank in ranks)
        existing = self._find_existing_group(ranks, supported_task_kinds)
        if existing is not None:
            return existing
        sp_size = len(ranks)
        group_id = f"edf_sp{sp_size}_r{'_'.join(str(rank) for rank in ranks)}"
        return self.topology.ensure_group(
            ExecutionGroupSpec(
                group_id=group_id,
                ranks=ranks,
                parallel_spec=ParallelSpec(tp=1, sp=sp_size, cfg=1),
                supported_task_kinds=supported_task_kinds,
                ulysses_degree=sp_size,
                ring_degree=1,
            )
        )

    def _find_existing_group(
        self,
        ranks: tuple[int, ...],
        supported_task_kinds: tuple[TaskKind, ...],
    ) -> ExecutionGroupSpec | None:
        required = set(supported_task_kinds)
        for group in self.topology.groups:
            if tuple(int(rank) for rank in group.ranks) != ranks:
                continue
            if int(group.parallel_spec.tp) != 1 or int(group.parallel_spec.cfg) != 1:
                continue
            if int(group.parallel_spec.sp) != len(ranks):
                continue
            if not required.issubset(set(group.supported_task_kinds)):
                continue
            return group
        return None

    def _validate_explicit_task_group(self, task: InferenceTask) -> None:
        group = self.topology.get_group(task.group_id or "")
        if task.kind not in group.supported_task_kinds:
            raise ValueError(
                f"task {task.task_id} kind={task.kind.value} is not supported by group {task.group_id!r}"
            )

    def _normalize_allowed_sp_sizes(self, allowed_sp_sizes: Iterable[int] | None) -> tuple[int, ...]:
        max_size = max(1, len(self.topology.workers))
        if allowed_sp_sizes is None:
            sizes: set[int] = {1}
            size = 2
            while size <= max_size:
                sizes.add(size)
                size *= 2
            sizes.add(max_size)
        else:
            sizes = {int(size) for size in allowed_sp_sizes}
            sizes.add(1)
        valid = sorted((size for size in sizes if 1 <= size <= max_size), reverse=True)
        if not valid:
            raise ValueError("edf_greedy allowed_sp_sizes must contain at least one valid size")
        return tuple(valid)

    def _request_sort_key(self, request_id: str) -> tuple[float, int, float, int, str]:
        meta = self.request_meta.get(request_id)
        if meta is None:
            return (math.inf, 0, 0.0, 0, request_id)
        return (meta.deadline_ms, -meta.priority, meta.arrival_ms, meta.order, request_id)

    def _meta_from_plan(self, plan: RequestExecutionPlan, order: int) -> _EdfGreedyRequestMeta:
        value = plan.initial_artifacts[0].value if plan.initial_artifacts else None
        task_priority = max((int(task.priority) for task in plan.tasks.values()), default=0)
        priority = int(self._metadata_number(value, ("runtime_v2_priority", "priority"), default=task_priority))
        arrival_ms = float(self._metadata_number(value, ("runtime_v2_arrival_ms", "arrival_ms"), default=0.0))
        raw_deadline_ms = float(
            self._metadata_number(
                value,
                ("runtime_v2_deadline_ms", "deadline_ms"),
                default=math.inf,
            )
        )
        # arrival_ms/deadline_ms arrive in the workload time base, but now_ms is
        # driven by worker-event monotonic-clock timestamps. Re-anchor the
        # deadline onto the monotonic clock at submission: the relative budget
        # (deadline - arrival) is epoch-independent, and for a replayed trace
        # submission time tracks arrival time so EDF ordering is preserved (a
        # constant shift). Without this, EdfBestFit's _estimate_finish_ms (in
        # monotonic ms) is compared against a workload-ms deadline and the
        # best-fit deadline test never holds.
        submit_ms = time.monotonic_ns() / 1_000_000.0
        self.now_ms = max(self.now_ms, submit_ms)
        if math.isfinite(raw_deadline_ms):
            deadline_ms = submit_ms + (raw_deadline_ms - arrival_ms)
        else:
            deadline_ms = math.inf
        # Diagnostic: shows whether per-request EDF metadata actually reached
        # the policy. raw_deadline_ms=inf or raw_arrival_ms=0 for every request
        # means the extra_args plumbing dropped it and EDF degrades to FCFS.
        logger.info(
            "runtime_v2 edf meta: request_id=%s order=%s priority=%s "
            "raw_deadline_ms=%s raw_arrival_ms=%.1f deadline_ms=%.1f",
            plan.request_id,
            order,
            priority,
            raw_deadline_ms,
            arrival_ms,
            deadline_ms,
        )
        return _EdfGreedyRequestMeta(
            deadline_ms=deadline_ms,
            priority=priority,
            arrival_ms=arrival_ms,
            order=order,
        )

    @classmethod
    def _metadata_number(
        cls,
        value: Any,
        names: tuple[str, ...],
        *,
        default: float,
    ) -> float:
        for raw in cls._metadata_values(value, names):
            if raw is None:
                continue
            try:
                return float(raw)
            except (TypeError, ValueError):
                continue
        return float(default)

    @staticmethod
    def _metadata_values(value: Any, names: tuple[str, ...]) -> Iterable[Any]:
        for name in names:
            if value is not None and hasattr(value, name):
                yield getattr(value, name)
        sampling_params = getattr(value, "sampling_params", None)
        extra_args = getattr(sampling_params, "extra_args", None)
        if isinstance(extra_args, Mapping):
            for name in names:
                if name in extra_args:
                    yield extra_args[name]
