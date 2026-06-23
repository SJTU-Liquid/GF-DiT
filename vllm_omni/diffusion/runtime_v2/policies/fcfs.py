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


class FCFSSchedulerPolicy(SchedulerPolicy):
    """FCFS admission with one active request per execution group."""

    def __init__(
        self,
        topology: RuntimeTopology,
        task_runtime_estimator: TaskRuntimeEstimator | None = None,
    ) -> None:
        self.topology = topology
        self.task_runtime_estimator = task_runtime_estimator or default_srtf_task_runtime_estimator
        self.ready_queue: deque[InferenceTask] = deque()
        self.active_request_by_group: dict[str, str] = {}
        self.pending_requests_by_group: dict[str, deque[str]] = {}
        self.blocked_tasks_by_request: dict[str, list[InferenceTask]] = {}
        self.request_group: dict[str, str] = {}
        self.group_backlog_work: dict[str, float] = {}
        self.request_remaining_work: dict[str, float] = {}
        self.task_work: dict[str, float] = {}
        self.request_task_ids: dict[str, set[str]] = {}
        self._group_order = {group.group_id: idx for idx, group in enumerate(self.topology.groups)}

    def on_request_submitted(self, plan: RequestExecutionPlan) -> Iterable[InferenceTask]:
        # Bind once at request ingress. All downstream tasks inherit the same
        # group_id so execution never crosses groups.
        selected_group_id = self._bind_request_group(plan)
        total_work = 0.0
        for task in plan.tasks.values():
            work = self._estimate_task_work(task, plan, self.topology.get_group(selected_group_id).parallel_spec)
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
            group_id = task.group_id

            active_request = self.active_request_by_group.get(group_id)
            if active_request is None:
                self.active_request_by_group[group_id] = task.request_id
                active_request = task.request_id

            if active_request == task.request_id:
                task.status = TaskStatus.READY
                self.ready_queue.append(task)
                continue

            queue_for_group = self.pending_requests_by_group.setdefault(group_id, deque())
            if task.request_id not in queue_for_group:
                queue_for_group.append(task.request_id)
            self.blocked_tasks_by_request.setdefault(task.request_id, []).append(task)

        return self._take_dispatchable_tasks()

    def on_worker_event(self, event: WorkerEvent) -> Iterable[InferenceTask]:
        if event.kind in (WorkerEventKind.TASK_LAUNCH_END, WorkerEventKind.TASK_EXEC_END):
            self._charge_completed_task(event.request_id, event.task_id)

        if event.kind in (WorkerEventKind.REQUEST_FINISHED, WorkerEventKind.REQUEST_FAILED):
            request_id = event.request_id
            # Fall back to the bound group if the event omits group_id, so the
            # active slot is always released and queued requests are promoted
            # (mirrors SRTF). Otherwise a group-less event would strand the group.
            group_id = event.group_id or self.request_group.get(request_id)
            if self.active_request_by_group.get(group_id) == request_id:
                del self.active_request_by_group[group_id]

            queue_for_group = self.pending_requests_by_group.get(group_id)
            if queue_for_group:
                while queue_for_group:
                    next_request_id = queue_for_group.popleft()
                    blocked_tasks = self.blocked_tasks_by_request.pop(next_request_id, [])
                    if not blocked_tasks:
                        continue
                    self.active_request_by_group[group_id] = next_request_id
                    for task in blocked_tasks:
                        task.status = TaskStatus.READY
                        self.ready_queue.append(task)
                    break
                if not queue_for_group:
                    self.pending_requests_by_group.pop(group_id, None)
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
                # Earliest-free heuristic:
                # choose the group with minimal (current backlog + this plan estimate).
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
            total += self._estimate_task_work(task, plan, parallel_spec)
        return total

    def _estimate_task_work(
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
        self.blocked_tasks_by_request.pop(request_id, None)

    def _take_dispatchable_tasks(self) -> list[InferenceTask]:
        dispatchable: list[InferenceTask] = []
        while self.ready_queue:
            task = self.ready_queue.popleft()
            task.status = TaskStatus.DISPATCHED
            dispatchable.append(task)
        return dispatchable
