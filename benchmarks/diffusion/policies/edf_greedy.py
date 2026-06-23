# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import os
from collections import deque
from dataclasses import dataclass
from typing import Iterable

from vllm_omni.diffusion.runtime_v2.policies.common import (
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
from vllm_omni.diffusion.runtime_v2.scheduler import (
    DynamicStepFCFSSchedulerPolicy,
    FCFSSchedulerPolicy,
    TaskRuntimeEstimator,
)
from vllm_omni.diffusion.runtime_v2.topology import RuntimeTopology


@dataclass(frozen=True)
class _EdfGreedyRequestMeta:
    deadline_ms: float | None
    priority: int
    arrival_ms: float
    order: int


class EdfGreedyPolicy(DynamicStepFCFSSchedulerPolicy):
    """Strict EDF + greedy free-pool SP placement, with dynamic group synthesis.

    Mirrors the production EdfGreedySchedulerPolicy in
    vllm_omni/diffusion/runtime_v2/scheduler.py. The simulator no longer
    needs the c84 / sliding topology enumeration: pass a bootstrap topology
    of one world group + per-rank SP1 lanes, and intermediate SP-k groups
    are synthesized on demand via topology.ensure_group(...).

    Algorithm per dispatch tick:

      1. free_ranks = workers minus (outstanding_per_rank | reshard_holds_by_rank)
      2. ready-not-inflight requests sorted by
         (deadline, -priority, arrival, submission_order, request_id)
      3. For each (request, head_task) in EDF order:
           - single-rank stages (TEXT_ENCODE / VAE_DECODE / FINALIZE):
             pick min(free_ranks) and ensure an SP1 group for it.
           - DiT stages (DIT_PREPARE / TIMESTEP_PREPARE / DIT_STEP_CHUNK):
             for sp_size in allowed_sp_sizes (desc),
                 ranks = sorted(free_ranks)[:sp_size]
                 ensure_group(ranks); dispatch.
         Decrement free_ranks by claimed ranks; continue.

    allowed_sp_sizes defaults to {1, 2, 4, ..., 2^k <= W} ∪ {W} like
    production. The factory below also accepts the EDF_GREEDY_SP_SIZES
    env var for ablations.
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
        # Track plans for cost-aware SP selection (mirror production policy).
        self.plans: dict[str, RequestExecutionPlan] = {}
        self.request_inflight: dict[str, int] = {}
        self.request_last_group: dict[str, str] = {}
        self._task_index: dict[str, InferenceTask] = {}
        self._request_seq = 0
        self.now_ms = 0.0
        self.allowed_sp_sizes = self._normalize_allowed_sp_sizes(allowed_sp_sizes)

    # ---- plan + lifecycle ----

    def build_sim_plan(self, *, request, default_builder):
        # Strip any pre-assigned group_id off every task so the policy picks
        # placement dynamically at dispatch (matching production). The default
        # builder stamps a single group_id on every task; we override here.
        plan = default_builder(request)
        for task in plan.tasks.values():
            task.group_id = None
        return plan

    def on_request_submitted(self, plan: RequestExecutionPlan):
        self._request_seq += 1
        value = plan.initial_artifacts[0].value if plan.initial_artifacts else None
        self.request_meta[plan.request_id] = self._meta_from_value(value, self._request_seq)
        self.plans[plan.request_id] = plan
        for task in plan.tasks.values():
            self._task_index[task.task_id] = task
        root_tasks = [task for task in plan.tasks.values() if not task.dependencies]
        return self.on_tasks_runnable(root_tasks)

    def on_tasks_runnable(self, tasks: Iterable[InferenceTask]):
        for task in tasks:
            task.status = TaskStatus.READY
            self._task_index[task.task_id] = task
            self.ready_by_request.setdefault(task.request_id, deque()).append(task)
        return self._take_dispatchable_tasks()

    def on_worker_event(self, event: WorkerEvent):
        self.now_ms = max(self.now_ms, float(event.timestamp_ns) / 1_000_000.0)
        if event.kind in (WorkerEventKind.TASK_EXEC_END, WorkerEventKind.TASK_FAILED):
            self._release_dispatched_task(event)
        if event.kind in (WorkerEventKind.REQUEST_FINISHED, WorkerEventKind.REQUEST_FAILED):
            self._drop_request(event.request_id)
        return self._take_dispatchable_tasks()

    # ---- scheduling core ----

    def _take_dispatchable_tasks(self):
        free_ranks = self._free_rank_set()
        ready: list[tuple[str, InferenceTask]] = []
        for request_id, queue in self.ready_by_request.items():
            if not queue:
                continue
            if self.request_inflight.get(request_id, 0) > 0:
                continue
            ready.append((request_id, queue[0]))
        ready.sort(key=lambda rt: self._sort_key(rt[0]))
        out: list[InferenceTask] = []
        for request_id, task in ready:
            placement = self._select_placement(task, free_ranks)
            if placement is None:
                continue
            group_id, ranks = placement
            self._dispatch(task, request_id, group_id, ranks)
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
                    f"task {task.task_id} pinned to {task.group_id!r} ranks={ranks!r}"
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
            if task.kind in group.supported_task_kinds:
                return group.group_id, (rank,)
        return None

    def _select_dit_group(
        self,
        task: InferenceTask,
        free_ranks: set[int],
    ) -> tuple[str, tuple[int, ...]] | None:
        if not free_ranks:
            return None
        fitting = [s for s in self.allowed_sp_sizes if s <= len(free_ranks)]
        if not fitting:
            return None
        # Default order: largest-first (original greedy behavior).
        candidates_order = sorted(fitting, reverse=True)
        # If we have a cost-model-backed runtime estimator, reorder by
        # estimated per-task cost (ascending) and tie-break by smaller sp.
        # Prevents picking a wider SP whose collective overhead makes it
        # slower than a smaller SP (e.g. Qwen Image S@512: SP1 155 ms,
        # SP4 192 ms / step). Mirrors the same fix in the production policy.
        plan = self.plans.get(task.request_id)
        if self.task_runtime_estimator is not None and plan is not None:
            seq_len = _resolve_seq_len(task, plan)
            stage = _resolve_request_stage(task)

            def _cost(sp: int) -> tuple[float, int]:
                spec = ParallelSpec(tp=1, sp=sp, cfg=1)
                c = self.task_runtime_estimator(spec, seq_len, stage)
                # Estimators (production & simulator) return 0.0 when the
                # cost model has no entry for this parallel_spec. Treat that
                # as "unknown" and sort it LAST so we don't accidentally
                # prefer a wider SP that has no cost-model coverage.
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

    def _pick_ranks_for_request(
        self,
        request_id: str,
        sp_size: int,
        free_ranks: set[int],
    ) -> tuple[int, ...]:
        # Mirror runtime EdfGreedySchedulerPolicy._pick_ranks_for_request:
        # prefer the request's last group when it is still entirely free at the
        # same SP width, so consecutive DiT chunks don't migrate ranks merely
        # because sorted(free)[:sp] happens to start at a different index.
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

    def _dispatch(self, task, request_id, group_id, ranks):
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

    def _release_dispatched_task(self, event):
        target = self.dispatched_kind.pop(event.task_id, None)
        ranks = self.dispatched_ranks_by_task.pop(event.task_id, ())
        if target is not None:
            remaining = self.outstanding_per_group.get(target, 0)
            if remaining <= 1:
                self.outstanding_per_group.pop(target, None)
            else:
                self.outstanding_per_group[target] = remaining - 1
            self._release_rank_counts(self.outstanding_per_rank, ranks)
        rid = event.request_id
        inflight = self.request_inflight.get(rid, 0)
        if inflight <= 1:
            self.request_inflight.pop(rid, None)
        else:
            self.request_inflight[rid] = inflight - 1
        if event.group_id:
            self.request_last_group[rid] = event.group_id

    def _drop_request(self, request_id):
        self.ready_by_request.pop(request_id, None)
        self.request_meta.pop(request_id, None)
        self.plans.pop(request_id, None)
        self.request_inflight.pop(request_id, None)
        self.request_last_group.pop(request_id, None)

    # ---- helpers ----

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

    def _normalize_allowed_sp_sizes(
        self,
        allowed_sp_sizes: Iterable[int] | None,
    ) -> tuple[int, ...]:
        max_size = max(1, len(self.topology.workers))
        if allowed_sp_sizes is None:
            sizes: set[int] = {1}
            size = 2
            while size <= max_size:
                sizes.add(size)
                size *= 2
            sizes.add(max_size)
        else:
            sizes = {int(s) for s in allowed_sp_sizes}
            sizes.add(1)
        valid = sorted((s for s in sizes if 1 <= s <= max_size), reverse=True)
        if not valid:
            raise ValueError("edf_greedy allowed_sp_sizes must contain at least one valid size")
        return tuple(valid)

    def _sort_key(self, request_id: str):
        meta = self.request_meta.get(request_id)
        if meta is None:
            return (math.inf, 0, 0.0, 0, request_id)
        deadline = meta.deadline_ms if meta.deadline_ms is not None else math.inf
        return (deadline, -meta.priority, meta.arrival_ms, meta.order, request_id)

    def _meta_from_value(self, value, order: int) -> _EdfGreedyRequestMeta:
        priority = int(getattr(value, "priority", 0) or 0)
        arrival_ms = float(getattr(value, "arrival_ms", 0.0) or 0.0)
        deadline_raw = getattr(value, "deadline_ms", None)
        deadline_ms = float(deadline_raw) if deadline_raw is not None else None
        self.now_ms = max(self.now_ms, arrival_ms)
        return _EdfGreedyRequestMeta(
            deadline_ms=deadline_ms,
            priority=priority,
            arrival_ms=arrival_ms,
            order=order,
        )


def make_edf_greedy(topology, task_runtime_estimator):
    raw = os.getenv("EDF_GREEDY_SP_SIZES", "").strip()
    allowed_sp_sizes: tuple[int, ...] | None = None
    if raw:
        allowed_sp_sizes = tuple(int(s.strip()) for s in raw.split(",") if s.strip())
    return EdfGreedyPolicy(
        topology=topology,
        task_runtime_estimator=task_runtime_estimator,
        allowed_sp_sizes=allowed_sp_sizes,
    )
