# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from collections import deque
from collections.abc import Iterable

from vllm.logger import init_logger

from vllm_omni.diffusion.runtime_v2.interfaces import SchedulerPolicy
from vllm_omni.diffusion.runtime_v2.policies.common import (
    TaskRuntimeEstimator,
    _candidate_groups_for_plan,
    _resolve_explicit_request_group,
    _resolve_request_stage,
    _resolve_seq_len,
    default_srtf_task_runtime_estimator,
)
from vllm_omni.diffusion.runtime_v2.protocol import (
    InferenceTask,
    ParallelSpec,
    RequestExecutionPlan,
    TaskStatus,
    WorkerEvent,
    WorkerEventKind,
)
from vllm_omni.diffusion.runtime_v2.topology import RuntimeTopology

logger = init_logger(__name__)


class SRTFSchedulerPolicy(SchedulerPolicy):
    """Preemptive SRTF policy at task boundaries with pluggable runtime estimate."""

    def __init__(
        self,
        topology: RuntimeTopology,
        task_runtime_estimator: TaskRuntimeEstimator | None = None,
    ) -> None:
        self.topology = topology
        self.task_runtime_estimator = task_runtime_estimator or default_srtf_task_runtime_estimator
        # group_id -> request_id -> FIFO runnable task queue.
        self.runnable_by_group: dict[str, dict[str, deque[InferenceTask]]] = {}
        # Active request chosen for each group (tie-breaking only).
        self.active_request_by_group: dict[str, str] = {}
        # request_id -> group_id.
        self.request_group: dict[str, str] = {}
        # request_id -> remaining estimated runtime.
        self.request_remaining_work: dict[str, float] = {}
        # group_id -> remaining estimated runtime for all assigned requests.
        self.group_backlog_work: dict[str, float] = {}
        # task_id -> estimated runtime, charged once at TASK_LAUNCH_END.
        self.task_work: dict[str, float] = {}
        # request_id -> task ids not yet charged/cleaned.
        self.request_task_ids: dict[str, set[str]] = {}
        # Outstanding throttling:
        # outstanding = running + queued-on-worker.
        self.max_queue_ahead_per_group = 1
        self.outstanding_tasks_by_group: dict[str, int] = {}
        # Launch throttling:
        # tasks dispatched to worker but not launch-finished yet.
        self.launch_pending_by_group: dict[str, int] = {}
        self.tasks_pending_launch: set[str] = set()
        self.dispatched_task_group: dict[str, str] = {}
        self._group_order = {group.group_id: idx for idx, group in enumerate(self.topology.groups)}
        # Monotonic submission order, used as the final tie-break so that requests
        # with equal estimated work are served FIFO (by arrival) rather than by the
        # lexicographic order of a random UUID request_id (which starves whichever
        # request keeps losing the UUID comparison).
        self._submit_counter = 0
        self.request_submit_seq: dict[str, int] = {}

    def on_request_submitted(self, plan: RequestExecutionPlan) -> Iterable[InferenceTask]:
        # Same request-level binding semantics as FCFS: group is chosen once
        # before any runnable task enters SRTF queues.
        if plan.request_id not in self.request_submit_seq:
            self.request_submit_seq[plan.request_id] = self._submit_counter
            self._submit_counter += 1
        selected_group_id = self._bind_request_group(plan)
        total_work = 0.0
        for task in plan.tasks.values():
            work = self._estimate_task_work(task, plan)
            self.task_work[task.task_id] = work
            total_work += work
        self.request_task_ids[plan.request_id] = set(plan.tasks.keys())
        self.request_remaining_work[plan.request_id] = total_work
        self.group_backlog_work[selected_group_id] = self.group_backlog_work.get(selected_group_id, 0.0) + total_work
        root_tasks = [task for task in plan.tasks.values() if not task.dependencies]
        return self.on_tasks_runnable(root_tasks)

    def on_tasks_runnable(self, tasks: Iterable[InferenceTask]) -> Iterable[InferenceTask]:
        for task in tasks:
            if task.group_id is None:
                task.group_id = self.request_group.get(task.request_id)
            if task.group_id is None:
                raise ValueError(
                    f"request {task.request_id} has runnable task without bound group_id: {task.task_id}"
                )
            req_group = self.request_group.setdefault(task.request_id, task.group_id)
            if req_group != task.group_id:
                raise ValueError(
                    f"request {task.request_id} attempted to use multiple groups: "
                    f"{req_group!r} and {task.group_id!r}"
                )
            by_request = self.runnable_by_group.setdefault(task.group_id, {})
            queue = by_request.setdefault(task.request_id, deque())
            task.status = TaskStatus.READY
            queue.append(task)
        return self._take_dispatchable_tasks()

    def on_worker_event(self, event: WorkerEvent) -> Iterable[InferenceTask]:
        if event.kind in (
            WorkerEventKind.TASK_LAUNCH_END,
            WorkerEventKind.TASK_EXEC_END,
            WorkerEventKind.TASK_FAILED,
        ):
            self._mark_task_launch_finished(event.task_id)

        if event.kind in (
            WorkerEventKind.TASK_EXEC_END,
            WorkerEventKind.TASK_FAILED,
        ):
            self._release_dispatched_task(event.task_id)

        if event.kind == WorkerEventKind.TASK_LAUNCH_END:
            self._charge_completed_task(event.request_id, event.task_id)
            return []

        if event.kind == WorkerEventKind.TASK_EXEC_END:
            self._charge_completed_task(event.request_id, event.task_id)

        if event.kind in (
            WorkerEventKind.REQUEST_FINISHED,
            WorkerEventKind.REQUEST_FAILED,
        ):
            request_id = event.request_id
            group_id = event.group_id or self.request_group.get(request_id)
            if group_id is not None:
                by_request = self.runnable_by_group.get(group_id)
                if by_request is not None:
                    by_request.pop(request_id, None)
                    if not by_request:
                        self.runnable_by_group.pop(group_id, None)
                if self.active_request_by_group.get(group_id) == request_id:
                    self.active_request_by_group.pop(group_id, None)
            self._cleanup_request_work(request_id=request_id, group_id=group_id)
        return self._take_dispatchable_tasks()

    def _bind_request_group(self, plan: RequestExecutionPlan) -> str:
        explicit_group_id = _resolve_explicit_request_group(plan)
        candidate_group_ids = _candidate_groups_for_plan(self.topology, plan)
        if explicit_group_id is not None:
            if explicit_group_id not in candidate_group_ids:
                raise ValueError(
                    f"request {plan.request_id} is pinned to group {explicit_group_id!r}, "
                    "but that group does not support all task kinds in the plan"
                )
            selected_group_id = explicit_group_id
        else:
            if not candidate_group_ids:
                task_kinds = sorted({task.kind.value for task in plan.tasks.values()})
                raise ValueError(f"no execution group supports all request task kinds: {task_kinds!r}")
            if len(candidate_group_ids) == 1:
                selected_group_id = candidate_group_ids[0]
            else:
                # Earliest-free heuristic with stable tie-breakers.
                selected_group_id = min(
                    candidate_group_ids,
                    key=lambda group_id: (
                        self.group_backlog_work.get(group_id, 0.0)
                        + self._estimate_plan_work(plan, self.topology.get_group(group_id).parallel_spec),
                        self._group_order.get(group_id, len(self._group_order)),
                        group_id,
                    ),
                )
        self.request_group[plan.request_id] = selected_group_id
        for task in plan.tasks.values():
            if task.group_id is None:
                task.group_id = selected_group_id
            elif task.group_id != selected_group_id:
                raise ValueError(
                    f"request {plan.request_id} contains mismatched task group_id={task.group_id!r}; "
                    f"expected {selected_group_id!r}"
                )
        return selected_group_id

    def _estimate_plan_work(self, plan: RequestExecutionPlan, parallel_spec: ParallelSpec) -> float:
        total = 0.0
        for task in plan.tasks.values():
            total += self._estimate_task_work_with_parallel(task, plan, parallel_spec)
        return total

    def _charge_completed_task(self, request_id: str, task_id: str) -> None:
        remaining = self.request_remaining_work.get(request_id)
        work = self.task_work.pop(task_id, None)
        request_tasks = self.request_task_ids.get(request_id)
        if request_tasks is not None:
            request_tasks.discard(task_id)
        if remaining is None or work is None:
            return
        new_remaining = max(0.0, remaining - work)
        charged = remaining - new_remaining
        if new_remaining <= 0:
            self.request_remaining_work.pop(request_id, None)
        else:
            self.request_remaining_work[request_id] = new_remaining
        group_id = self.request_group.get(request_id)
        if group_id is not None and charged > 0:
            backlog = self.group_backlog_work.get(group_id, 0.0) - charged
            if backlog <= 1e-9:
                self.group_backlog_work.pop(group_id, None)
            else:
                self.group_backlog_work[group_id] = backlog

    def _cleanup_request_work(self, request_id: str, group_id: str | None = None) -> None:
        self.request_submit_seq.pop(request_id, None)
        task_ids = self.request_task_ids.pop(request_id, None)
        if task_ids:
            for task_id in task_ids:
                self.task_work.pop(task_id, None)
        remaining = self.request_remaining_work.pop(request_id, 0.0)
        bound_group = self.request_group.pop(request_id, None)
        final_group_id = group_id or bound_group
        if final_group_id is not None and remaining > 0:
            backlog = self.group_backlog_work.get(final_group_id, 0.0) - remaining
            if backlog <= 1e-9:
                self.group_backlog_work.pop(final_group_id, None)
            else:
                self.group_backlog_work[final_group_id] = backlog

    def _release_dispatched_task(self, task_id: str) -> None:
        group_id = self.dispatched_task_group.pop(task_id, None)
        if group_id is None:
            return
        outstanding = self.outstanding_tasks_by_group.get(group_id, 0)
        if outstanding <= 1:
            self.outstanding_tasks_by_group.pop(group_id, None)
        else:
            self.outstanding_tasks_by_group[group_id] = outstanding - 1

    def _mark_task_launch_finished(self, task_id: str) -> None:
        if task_id not in self.tasks_pending_launch:
            return
        self.tasks_pending_launch.remove(task_id)
        group_id = self.dispatched_task_group.get(task_id)
        if group_id is None:
            return
        pending = self.launch_pending_by_group.get(group_id, 0)
        if pending <= 1:
            self.launch_pending_by_group.pop(group_id, None)
        else:
            self.launch_pending_by_group[group_id] = pending - 1

    def _take_dispatchable_tasks(self) -> list[InferenceTask]:
        dispatchable: list[InferenceTask] = []
        for group in self.topology.groups:
            group_id = group.group_id
            if self.launch_pending_by_group.get(group_id, 0) > 0:
                continue
            max_outstanding = 1 + self.max_queue_ahead_per_group
            if self.outstanding_tasks_by_group.get(group_id, 0) >= max_outstanding:
                continue
            by_request = self.runnable_by_group.get(group_id)
            if not by_request:
                continue
            request_id = self._select_shortest_request(group_id, by_request)
            if request_id is None:
                continue
            queue = by_request[request_id]
            task = queue.popleft()
            if not queue:
                del by_request[request_id]
                if not by_request:
                    self.runnable_by_group.pop(group_id, None)
            self.active_request_by_group[group_id] = request_id
            task.status = TaskStatus.DISPATCHED
            self.outstanding_tasks_by_group[group_id] = self.outstanding_tasks_by_group.get(group_id, 0) + 1
            self.dispatched_task_group[task.task_id] = group_id
            self.tasks_pending_launch.add(task.task_id)
            self.launch_pending_by_group[group_id] = self.launch_pending_by_group.get(group_id, 0) + 1
            dispatchable.append(task)
        return dispatchable

    def _select_shortest_request(
        self,
        group_id: str,
        by_request: dict[str, deque[InferenceTask]],
    ) -> str | None:
        if not by_request:
            return None
        active_request = self.active_request_by_group.get(group_id)
        return min(
            by_request.keys(),
            key=lambda request_id: (
                self.request_remaining_work.get(request_id, float("inf")),
                0 if request_id == active_request else 1,
                self.request_submit_seq.get(request_id, self._submit_counter),
                request_id,
            ),
        )

    def _estimate_task_work_with_parallel(
        self,
        task: InferenceTask,
        plan: RequestExecutionPlan,
        parallel_spec: ParallelSpec,
    ) -> float:
        seq_len = _resolve_seq_len(task, plan)
        request_stage = _resolve_request_stage(task)
        estimate = float(self.task_runtime_estimator(parallel_spec, seq_len, request_stage))
        if not math.isfinite(estimate):
            raise ValueError(
                f"invalid task runtime estimate for task_id={task.task_id}: {estimate} (must be finite number)"
            )
        if estimate < 0:
            raise ValueError(
                f"invalid task runtime estimate for task_id={task.task_id}: {estimate} (must be >= 0)"
            )
        return estimate

    def _estimate_task_work(self, task: InferenceTask, plan: RequestExecutionPlan) -> float:
        parallel_spec = self._resolve_parallel_spec(task)
        return self._estimate_task_work_with_parallel(task, plan, parallel_spec)

    def _resolve_parallel_spec(self, task: InferenceTask) -> ParallelSpec:
        if task.group_id is None:
            raise ValueError(f"task {task.task_id} has no bound group_id during runtime estimation")
        return self.topology.get_group(task.group_id).parallel_spec

    @staticmethod
    def _resolve_seq_len(task: InferenceTask, plan: RequestExecutionPlan) -> int:
        return _resolve_seq_len(task, plan)

    @staticmethod
    def _resolve_request_stage(task: InferenceTask) -> str:
        return _resolve_request_stage(task)
