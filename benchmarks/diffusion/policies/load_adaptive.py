# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import os
from collections import deque
from dataclasses import dataclass
from typing import Iterable

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


from benchmarks.diffusion.policies.common import (
    _SingleGroupPolicyBase,
    _request_class_from_value,
    _request_deadline_ms,
    _request_priority,
    _groups_with_sp,
)

class LoadAdaptivePolicy(_SingleGroupPolicyBase):
    """Pick DiT group at admission based on current load.

    Low load: pick a latency group (largest SP available).
    High load: pick a throughput group (smallest SP available).
    """

    def __init__(
        self,
        topology: RuntimeTopology,
        task_runtime_estimator: TaskRuntimeEstimator | None = None,
        *,
        high_load_interarrival_ms: float = 120_000.0,
        high_load_cooldown_ms: float = 60_000.0,
        backlog_high_load_requests: int = 2,
    ) -> None:
        super().__init__(topology=topology, task_runtime_estimator=task_runtime_estimator)
        self.high_load_interarrival_ms = float(high_load_interarrival_ms)
        self.high_load_cooldown_ms = float(high_load_cooldown_ms)
        self.backlog_high_load_requests = int(backlog_high_load_requests)
        max_sp = max(int(g.parallel_spec.sp) for g in topology.groups)
        min_sp = min(int(g.parallel_spec.sp) for g in topology.groups)
        self._latency_group_ids = _groups_with_sp(topology, max_sp)
        self._throughput_group_ids = _groups_with_sp(topology, min_sp)
        if not self._latency_group_ids or not self._throughput_group_ids:
            raise ValueError("LoadAdaptivePolicy needs both latency and throughput lanes")
        self._last_arrival_ms: float | None = None
        self._high_load_until_ms = -math.inf
        self._inflight_requests: set[str] = set()
        self._latency_cursor = 0
        self._throughput_cursor = 0

    def _is_high_load(self, plan: RequestExecutionPlan) -> bool:
        value = plan.initial_artifacts[0].value if plan.initial_artifacts else None
        arrival_ms = float(getattr(value, "arrival_ms", 0.0) or 0.0)
        if self._last_arrival_ms is not None:
            interarrival = arrival_ms - self._last_arrival_ms
            if interarrival <= self.high_load_interarrival_ms:
                self._high_load_until_ms = max(
                    self._high_load_until_ms,
                    arrival_ms + self.high_load_cooldown_ms,
                )
        self._last_arrival_ms = arrival_ms
        if arrival_ms <= self._high_load_until_ms:
            return True
        return len(self._inflight_requests) >= self.backlog_high_load_requests

    def _pick_dit_group_id(self, plan: RequestExecutionPlan) -> str:
        high_load = self._is_high_load(plan)
        self._inflight_requests.add(plan.request_id)
        if high_load:
            group_id = self._throughput_group_ids[self._throughput_cursor % len(self._throughput_group_ids)]
            self._throughput_cursor += 1
            return group_id
        group_id = self._latency_group_ids[self._latency_cursor % len(self._latency_group_ids)]
        self._latency_cursor += 1
        return group_id

    def on_worker_event(self, event: WorkerEvent) -> Iterable[InferenceTask]:
        if event.kind in (WorkerEventKind.REQUEST_FINISHED, WorkerEventKind.REQUEST_FAILED):
            self._inflight_requests.discard(event.request_id)
        return super().on_worker_event(event)


class PriorityLoadAdaptivePolicy(LoadAdaptivePolicy):
    """LoadAdaptivePolicy with priority-aware request activation.

    FCFSSchedulerPolicy enforces one active request per group. When the
    active request finishes, the parent class pops the next request from
    ``pending_requests_by_group`` in FIFO order. This subclass reorders
    that pending queue by (priority desc, deadline asc, submission asc)
    before each activation. It also reorders the shared ``ready_queue`` so
    inter-group dispatch order respects priority when multiple groups
    have ready tasks.
    """

    def __init__(self, topology, task_runtime_estimator=None, **kwargs):
        super().__init__(topology=topology, task_runtime_estimator=task_runtime_estimator, **kwargs)
        self._request_priority: dict[str, int] = {}
        self._request_deadline: dict[str, float] = {}
        self._request_order: dict[str, int] = {}
        self._request_seq = 0

    def on_request_submitted(self, plan: RequestExecutionPlan) -> Iterable[InferenceTask]:
        self._request_seq += 1
        self._request_order[plan.request_id] = self._request_seq
        self._request_priority[plan.request_id] = _request_priority(plan)
        deadline = _request_deadline_ms(plan)
        self._request_deadline[plan.request_id] = deadline if deadline is not None else math.inf
        return super().on_request_submitted(plan)

    def on_tasks_runnable(self, tasks: Iterable[InferenceTask]) -> Iterable[InferenceTask]:
        # Reorder per-group pending queues first so the next activation
        # chooses the highest-priority queued request.
        self._reorder_pending_queues()
        result = super().on_tasks_runnable(tasks)
        # Reorder the shared ready_queue too, in case multiple groups'
        # tasks are pending dispatch.
        ordered = sorted(self.ready_queue, key=self._task_sort_key)
        self.ready_queue.clear()
        self.ready_queue.extend(ordered)
        return result

    def on_worker_event(self, event):
        # Same reorder before the parent's REQUEST_FINISHED handler activates
        # a new request from pending_requests_by_group.
        self._reorder_pending_queues()
        return super().on_worker_event(event)

    def _reorder_pending_queues(self) -> None:
        for group_id, queue in list(self.pending_requests_by_group.items()):
            if len(queue) <= 1:
                continue
            ordered = sorted(queue, key=self._request_sort_key)
            queue.clear()
            queue.extend(ordered)

    def _request_sort_key(self, request_id: str):
        priority = self._request_priority.get(request_id, 0)
        deadline = self._request_deadline.get(request_id, math.inf)
        order = self._request_order.get(request_id, 0)
        return (-priority, deadline, order, request_id)

    def _task_sort_key(self, task: InferenceTask):
        priority = self._request_priority.get(task.request_id, 0)
        deadline = self._request_deadline.get(task.request_id, math.inf)
        order = self._request_order.get(task.request_id, 0)
        return (-priority, deadline, order, task.task_id)


def make_load_adaptive(topology, task_runtime_estimator):
    return LoadAdaptivePolicy(
        topology=topology,
        task_runtime_estimator=task_runtime_estimator,
        high_load_interarrival_ms=float(os.getenv("ADAPTIVE_SP_HIGH_LOAD_INTERARRIVAL_MS", "120000")),
        backlog_high_load_requests=int(os.getenv("ADAPTIVE_SP_BACKLOG_HIGH_LOAD_REQUESTS", "2")),
        high_load_cooldown_ms=float(os.getenv("ADAPTIVE_SP_HIGH_LOAD_COOLDOWN_MS", "60000")),
    )


def make_priority_load_adaptive(topology, task_runtime_estimator):
    return PriorityLoadAdaptivePolicy(
        topology=topology,
        task_runtime_estimator=task_runtime_estimator,
        high_load_interarrival_ms=float(os.getenv("ADAPTIVE_SP_HIGH_LOAD_INTERARRIVAL_MS", "120000")),
        backlog_high_load_requests=int(os.getenv("ADAPTIVE_SP_BACKLOG_HIGH_LOAD_REQUESTS", "2")),
        high_load_cooldown_ms=float(os.getenv("ADAPTIVE_SP_HIGH_LOAD_COOLDOWN_MS", "60000")),
    )
