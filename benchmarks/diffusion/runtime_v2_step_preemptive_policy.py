#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Step-boundary preemptive elastic policies for runtime_v2 simulations.

These policies deliberately live outside ``runtime_v2_policy_simulator.py``.
The workload remains placement-agnostic: request JSON describes request shape
and priority, while this policy chooses concrete execution groups at runnable
task boundaries.
"""

from __future__ import annotations

import math
import os
from collections import deque
from dataclasses import dataclass
from typing import Iterable

from vllm_omni.diffusion.runtime_v2.protocol import (
    InferenceTask,
    RequestExecutionPlan,
    TaskKind,
    TaskStatus,
    WorkerEvent,
    WorkerEventKind,
)
from vllm_omni.diffusion.runtime_v2.scheduler import (
    DynamicStepFCFSSchedulerPolicy,
    TaskRuntimeEstimator,
)
from vllm_omni.diffusion.runtime_v2.topology import RuntimeTopology


@dataclass(frozen=True)
class RequestMeta:
    request_id: str
    request_class: str
    priority: int
    arrival_ms: float
    preemptible: bool
    deadline_ms: float | None
    profiled_optimal_latency_ms: float | None


@dataclass(frozen=True)
class ActionCandidate:
    task: InferenceTask
    request_id: str
    group_id: str
    action: str
    score: float
    rank_mask: int
    claimed_ranks: tuple[int, ...]
    sp_degree: int
    estimated_task_ms: float
    estimated_remaining_ms: float
    reshard_cost_ms: float
    source_group_id: str | None


class StepBoundaryPreemptiveElasticPolicy(DynamicStepFCFSSchedulerPolicy):
    """Rank-mask DP scheduler with cooperative step-boundary preemption.

    The simulator marks dependencies runnable at task launch-end, but this
    policy only dispatches the next task for a request after the previous task
    has reached TASK_EXEC_END.  That keeps DiT step boundaries cooperative even
    when the next group is rank-disjoint from the previous group.
    """

    def __init__(
        self,
        topology: RuntimeTopology,
        task_runtime_estimator: TaskRuntimeEstimator | None = None,
        *,
        mode: str,
        reshard_ms: float = 0.0,
        top_k: int = 6,
        demote_max_sp: int = 2,
        max_pause_ms: float = 120_000.0,
    ) -> None:
        super().__init__(topology=topology)
        if mode not in {"pause", "demote"}:
            raise ValueError(f"mode must be pause or demote, got {mode!r}")
        self.mode = mode
        self.task_runtime_estimator = task_runtime_estimator
        self.reshard_ms = max(0.0, float(reshard_ms))
        self.top_k = max(1, int(top_k))
        self.demote_max_sp = max(1, int(demote_max_sp))
        self.max_pause_ms = max(0.0, float(max_pause_ms))

        self.ready_by_request: dict[str, deque[InferenceTask]] = {}
        self.request_meta: dict[str, RequestMeta] = {}
        self.request_order: dict[str, int] = {}
        self.request_last_group: dict[str, str] = {}
        self.request_inflight: dict[str, int] = {}
        self.completed_steps: dict[str, int] = {}
        self.paused_since_ms: dict[str, float] = {}
        self.pause_total_ms: dict[str, float] = {}
        self.preempt_requested_ms: dict[str, float] = {}
        self.virtual_queues: dict[str, float] = {"S": 0.0, "M": 0.0, "L": 0.0}
        self.now_ms = 0.0
        self._request_seq = 0
        self._task_index: dict[str, InferenceTask] = {}
        self._claim_ranks_by_task: dict[str, tuple[int, ...]] = {}

        self._group_sp = {group.group_id: int(group.parallel_spec.sp) for group in topology.groups}
        self._group_rank_masks = {
            group.group_id: _rank_mask(tuple(group.ranks))
            for group in topology.groups
        }
        self._max_sp = max(self._group_sp.values())
        self._min_sp = min(self._group_sp.values())
        self._latency_group_ids = tuple(
            group.group_id for group in topology.groups if self._group_sp[group.group_id] == self._max_sp
        )
        self._small_group_ids = tuple(
            group.group_id for group in topology.groups if self._group_sp[group.group_id] <= self.demote_max_sp
        ) or tuple(
            group.group_id for group in topology.groups if self._group_sp[group.group_id] == self._min_sp
        )

    def on_request_submitted(self, plan: RequestExecutionPlan) -> Iterable[InferenceTask]:
        meta = self._meta_from_plan(plan)
        self.now_ms = max(self.now_ms, meta.arrival_ms)
        self.request_meta[plan.request_id] = meta
        self._request_seq += 1
        self.request_order[plan.request_id] = self._request_seq
        if meta.request_class == "S":
            self._mark_preempt_requested(caused_by=plan.request_id)
        root_tasks = [task for task in plan.tasks.values() if not task.dependencies]
        return self.on_tasks_runnable(root_tasks)

    def on_tasks_runnable(self, tasks: Iterable[InferenceTask]) -> Iterable[InferenceTask]:
        for task in tasks:
            task.status = TaskStatus.READY
            self._task_index[task.task_id] = task
            self.ready_by_request.setdefault(task.request_id, deque()).append(task)
        return self._take_dispatchable_tasks()

    def on_worker_event(self, event: WorkerEvent) -> Iterable[InferenceTask]:
        self.now_ms = max(self.now_ms, float(event.timestamp_ns) / 1_000_000.0)
        if event.kind in (WorkerEventKind.TASK_EXEC_END, WorkerEventKind.TASK_FAILED):
            self._release_dispatched_task(event)
        if event.kind in (WorkerEventKind.REQUEST_FINISHED, WorkerEventKind.REQUEST_FAILED):
            self._update_virtual_queue(event.request_id)
            self._drop_request(event.request_id)
        return self._take_dispatchable_tasks()

    def _take_dispatchable_tasks(self) -> list[InferenceTask]:
        candidates_by_request: list[tuple[str, list[ActionCandidate]]] = []
        for request_id in self._ready_request_ids():
            if self.request_inflight.get(request_id, 0) > 0:
                continue
            queue = self.ready_by_request.get(request_id)
            if not queue:
                continue
            task = queue[0]
            if self._should_pause(request_id, task):
                continue
            candidates = self._candidate_actions(request_id, task)
            if candidates:
                candidates_by_request.append((request_id, candidates[: self.top_k]))

        selected = self._solve_rank_dp(candidates_by_request)
        out: list[InferenceTask] = []
        for candidate in selected:
            queue = self.ready_by_request.get(candidate.request_id)
            if not queue or queue[0].task_id != candidate.task.task_id:
                continue
            queue.popleft()
            if not queue:
                self.ready_by_request.pop(candidate.request_id, None)
            self._resume(candidate.request_id)
            self._dispatch_candidate(candidate, out)
        return out

    def _dispatch_candidate(self, candidate: ActionCandidate, out: list[InferenceTask]) -> None:
        task = candidate.task
        task.group_id = candidate.group_id
        task.status = TaskStatus.DISPATCHED
        self._claim_ranks(candidate.claimed_ranks, reshard=False)
        self.dispatched_kind[task.task_id] = candidate.group_id
        self.dispatched_ranks_by_task[task.task_id] = candidate.claimed_ranks
        self._claim_ranks_by_task[task.task_id] = candidate.claimed_ranks
        self.outstanding_per_group[candidate.group_id] = self.outstanding_per_group.get(candidate.group_id, 0) + 1
        self.request_inflight[task.request_id] = self.request_inflight.get(task.request_id, 0) + 1
        out.append(task)

    def _release_dispatched_task(self, event: WorkerEvent) -> None:
        task = self._task_index.get(event.task_id)
        target = self.dispatched_kind.pop(event.task_id, None)
        ranks = self.dispatched_ranks_by_task.pop(event.task_id, ())
        self._claim_ranks_by_task.pop(event.task_id, None)
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
        if task is not None:
            self.request_last_group[request_id] = event.group_id
            if task.kind == TaskKind.DIT_STEP_CHUNK and task.step_range is not None:
                self.completed_steps[request_id] = int(task.step_range.end)

    def _candidate_actions(self, request_id: str, task: InferenceTask) -> list[ActionCandidate]:
        out: list[ActionCandidate] = []
        source_group_id = self.request_last_group.get(request_id)
        for group_id in self._candidate_group_ids(request_id, task):
            group = self.topology.get_group(group_id)
            if task.kind not in group.supported_task_kinds:
                continue
            target_ranks = tuple(group.ranks)
            claimed_ranks = target_ranks
            reshard_cost_ms = 0.0
            if source_group_id is not None and source_group_id != group_id:
                claimed_ranks = tuple(sorted(set(target_ranks) | set(self.topology.get_group(source_group_id).ranks)))
                reshard_cost_ms = self.reshard_ms
            if not self._can_dispatch_ranks(claimed_ranks):
                continue
            task_ms = self._estimate_task_ms(task, group_id)
            remaining_ms = self._estimate_remaining_ms(task, group_id)
            action = self._action_name(source_group_id, group_id)
            score = self._score_action(
                request_id=request_id,
                task=task,
                group_id=group_id,
                action=action,
                task_ms=task_ms,
                remaining_ms=remaining_ms,
                reshard_cost_ms=reshard_cost_ms,
            )
            out.append(
                ActionCandidate(
                    task=task,
                    request_id=request_id,
                    group_id=group_id,
                    action=action,
                    score=score,
                    rank_mask=_rank_mask(claimed_ranks),
                    claimed_ranks=claimed_ranks,
                    sp_degree=self._group_sp[group_id],
                    estimated_task_ms=task_ms,
                    estimated_remaining_ms=remaining_ms,
                    reshard_cost_ms=reshard_cost_ms,
                    source_group_id=source_group_id,
                )
            )
        out.sort(key=lambda item: (-item.score, item.reshard_cost_ms, -item.sp_degree, item.group_id))
        return out

    def _candidate_group_ids(self, request_id: str, task: InferenceTask) -> tuple[str, ...]:
        meta = self.request_meta[request_id]
        source = self.request_last_group.get(request_id)
        pressure = self._foreground_pressure()
        ordered: list[str] = []

        def add(items: Iterable[str]) -> None:
            for item in items:
                if item not in ordered:
                    ordered.append(item)

        if source is not None:
            add((source,))
        if meta.request_class == "S":
            add(self._latency_group_ids)
            add(group.group_id for group in self.topology.groups)
            return tuple(ordered)
        if meta.request_class == "L" and pressure and task.kind == TaskKind.DIT_STEP_CHUNK:
            if self.mode == "pause" and not self._pause_aged_out(request_id):
                return ()
            if self.mode == "demote":
                add(self._small_group_ids)
                if self._pause_aged_out(request_id):
                    add(group.group_id for group in self.topology.groups)
                return tuple(ordered)
        if meta.request_class == "M" and pressure:
            add(self._small_group_ids)
            add(self._latency_group_ids)
            return tuple(ordered)
        add(self._latency_group_ids)
        add(group.group_id for group in self.topology.groups)
        return tuple(ordered)

    def _score_action(
        self,
        *,
        request_id: str,
        task: InferenceTask,
        group_id: str,
        action: str,
        task_ms: float,
        remaining_ms: float,
        reshard_cost_ms: float,
    ) -> float:
        meta = self.request_meta[request_id]
        class_base = {"S": 100_000.0, "M": 10_000.0, "L": 1_000.0}.get(meta.request_class, 5_000.0)
        deadline_weight = {"S": 80_000.0, "M": 12_000.0, "L": 500.0}.get(meta.request_class, 1_000.0)
        progress_weight = {"S": 2_000.0, "M": 2_500.0, "L": 5_000.0}.get(meta.request_class, 1_000.0)
        reshard_weight = {"S": 4.0, "M": 6.0, "L": 10.0}.get(meta.request_class, 6.0)
        score = class_base
        score += progress_weight / (1.0 + task_ms / 1000.0)
        score -= reshard_weight * reshard_cost_ms
        score += 750.0 if action == "continue" else 0.0
        if meta.request_class == "L" and self._foreground_pressure():
            score += 6_000.0 if action == "demote" else 0.0
            score -= 20_000.0 if action == "promote" else 0.0
        paused_at = self.paused_since_ms.get(request_id)
        if paused_at is not None:
            score += min(15_000.0, max(0.0, self.now_ms - paused_at) * 0.25)
        if meta.deadline_ms is not None:
            slack = meta.deadline_ms - self.now_ms - remaining_ms - reshard_cost_ms
            scale = max(1.0, meta.profiled_optimal_latency_ms or remaining_ms or 1.0)
            score += deadline_weight * max(0.0, -slack) / scale
        score += self.virtual_queues.get(meta.request_class, 0.0) * 2_000.0
        if task.kind != TaskKind.DIT_STEP_CHUNK and meta.request_class == "L" and self._foreground_pressure():
            score -= 1_500.0
        return score

    def _solve_rank_dp(
        self,
        candidates_by_request: list[tuple[str, list[ActionCandidate]]],
    ) -> list[ActionCandidate]:
        dp: dict[int, tuple[float, list[ActionCandidate]]] = {0: (0.0, [])}
        for _request_id, candidates in candidates_by_request:
            next_dp = dict(dp)
            for used_mask, (value, actions) in dp.items():
                for candidate in candidates:
                    if used_mask & candidate.rank_mask:
                        continue
                    new_mask = used_mask | candidate.rank_mask
                    new_value = value + candidate.score
                    if new_mask not in next_dp or new_value > next_dp[new_mask][0]:
                        next_dp[new_mask] = (new_value, actions + [candidate])
            dp = next_dp
        best_value, best_actions = max(dp.values(), key=lambda item: (item[0], len(item[1])))
        return best_actions if best_value > 0.0 else []

    def _ready_request_ids(self) -> list[str]:
        return sorted(
            (request_id for request_id, queue in self.ready_by_request.items() if queue),
            key=self._request_sort_key,
        )

    def _request_sort_key(self, request_id: str) -> tuple[int, float, int, str]:
        meta = self.request_meta[request_id]
        deadline = meta.deadline_ms if meta.deadline_ms is not None else math.inf
        return (-meta.priority, deadline, self.request_order.get(request_id, 0), request_id)

    def _should_pause(self, request_id: str, task: InferenceTask) -> bool:
        meta = self.request_meta[request_id]
        if meta.request_class != "L" or not meta.preemptible:
            return False
        if task.kind != TaskKind.DIT_STEP_CHUNK or not self._foreground_pressure():
            return False
        if self.mode == "demote" and self._candidate_group_ids(request_id, task):
            return False
        if self._pause_aged_out(request_id):
            return False
        self.paused_since_ms.setdefault(request_id, self.now_ms)
        return True

    def _resume(self, request_id: str) -> None:
        paused_at = self.paused_since_ms.pop(request_id, None)
        if paused_at is not None:
            self.pause_total_ms[request_id] = self.pause_total_ms.get(request_id, 0.0) + max(0.0, self.now_ms - paused_at)

    def _pause_aged_out(self, request_id: str) -> bool:
        paused_at = self.paused_since_ms.get(request_id)
        return paused_at is not None and self.now_ms - paused_at >= self.max_pause_ms

    def _foreground_pressure(self) -> bool:
        for request_id, meta in self.request_meta.items():
            if meta.request_class != "S":
                continue
            if self.request_inflight.get(request_id, 0) > 0:
                return True
            if self.ready_by_request.get(request_id):
                return True
        return self.virtual_queues.get("S", 0.0) > 0.0

    def _mark_preempt_requested(self, *, caused_by: str) -> None:
        for request_id, meta in self.request_meta.items():
            if request_id == caused_by or meta.request_class != "L" or not meta.preemptible:
                continue
            if request_id in self.preempt_requested_ms:
                continue
            if request_id in self.request_last_group or self.request_inflight.get(request_id, 0) > 0:
                self.preempt_requested_ms[request_id] = self.now_ms

    def _update_virtual_queue(self, request_id: str) -> None:
        meta = self.request_meta.get(request_id)
        if meta is None or meta.deadline_ms is None:
            return
        epsilon = {"S": 0.01, "M": 0.05, "L": 0.20}.get(meta.request_class, 0.05)
        late = 1.0 if self.now_ms > meta.deadline_ms else 0.0
        self.virtual_queues[meta.request_class] = max(0.0, self.virtual_queues.get(meta.request_class, 0.0) + late - epsilon)

    def _drop_request(self, request_id: str) -> None:
        self.ready_by_request.pop(request_id, None)
        self.request_meta.pop(request_id, None)
        self.request_order.pop(request_id, None)
        self.request_last_group.pop(request_id, None)
        self.request_inflight.pop(request_id, None)
        self.completed_steps.pop(request_id, None)
        self.preempt_requested_ms.pop(request_id, None)
        paused_at = self.paused_since_ms.pop(request_id, None)
        if paused_at is not None:
            self.pause_total_ms[request_id] = self.pause_total_ms.get(request_id, 0.0) + max(0.0, self.now_ms - paused_at)

    def _estimate_task_ms(self, task: InferenceTask, group_id: str) -> float:
        if self.task_runtime_estimator is None:
            return 1.0
        group = self.topology.get_group(group_id)
        payload = dict(task.payload)
        seq_len = int(payload.get("seq_len", payload.get("seqlen", 1)))
        stage = str(payload.get("cost_model_stage", payload.get("request_stage", task.kind.value)))
        estimate = float(self.task_runtime_estimator(group.parallel_spec, seq_len, stage))
        if task.kind == TaskKind.DIT_STEP_CHUNK:
            features = dict(payload.get("sim_features", {}))
            chunk_steps = int(features.get("chunk_steps", 1))
            estimate *= max(1, chunk_steps)
        return max(0.0, estimate)

    def _estimate_remaining_ms(self, task: InferenceTask, group_id: str) -> float:
        metadata = dict(task.payload.get("request_metadata", {}))
        num_steps = int(metadata.get("num_steps", 1))
        completed = self.completed_steps.get(task.request_id, 0)
        if task.kind == TaskKind.DIT_STEP_CHUNK and task.step_range is not None:
            completed = max(completed, int(task.step_range.start))
            current_steps = task.step_range.end - task.step_range.start
        else:
            current_steps = 0
        task_ms = self._estimate_task_ms(task, group_id)
        if task.kind != TaskKind.DIT_STEP_CHUNK:
            return task_ms
        per_step_ms = task_ms / max(1, current_steps)
        remaining_after_current = max(0, num_steps - completed - current_steps)
        return task_ms + per_step_ms * remaining_after_current

    def _action_name(self, source_group_id: str | None, target_group_id: str) -> str:
        if source_group_id is None or source_group_id == target_group_id:
            return "continue"
        source_sp = self._group_sp[source_group_id]
        target_sp = self._group_sp[target_group_id]
        if target_sp < source_sp:
            return "demote"
        if target_sp > source_sp:
            return "promote"
        return "migrate"

    def _meta_from_plan(self, plan: RequestExecutionPlan) -> RequestMeta:
        value = plan.initial_artifacts[0].value if plan.initial_artifacts else None
        request_id = plan.request_id
        request_class = _request_class(request_id)
        priority = int(getattr(value, "priority", 0) or _default_priority(request_class))
        arrival_ms = float(getattr(value, "arrival_ms", 0.0) or 0.0)
        deadline_ms = _float_attr(value, "deadline_ms")
        profiled_optimal_latency_ms = _float_attr(value, "profiled_optimal_latency_ms")
        return RequestMeta(
            request_id=request_id,
            request_class=request_class,
            priority=priority,
            arrival_ms=arrival_ms,
            preemptible=request_class == "L",
            deadline_ms=deadline_ms,
            profiled_optimal_latency_ms=profiled_optimal_latency_ms,
        )


def make_pause_policy(
    topology: RuntimeTopology,
    task_runtime_estimator: TaskRuntimeEstimator,
) -> StepBoundaryPreemptiveElasticPolicy:
    return StepBoundaryPreemptiveElasticPolicy(
        topology=topology,
        task_runtime_estimator=task_runtime_estimator,
        mode="pause",
        reshard_ms=float(os.getenv("STEP_PREEMPT_RESHARD_MS", "0")),
        top_k=int(os.getenv("STEP_PREEMPT_TOP_K", "6")),
        demote_max_sp=int(os.getenv("STEP_PREEMPT_DEMOTE_MAX_SP", "2")),
        max_pause_ms=float(os.getenv("STEP_PREEMPT_MAX_PAUSE_MS", "120000")),
    )


def make_demote_policy(
    topology: RuntimeTopology,
    task_runtime_estimator: TaskRuntimeEstimator,
) -> StepBoundaryPreemptiveElasticPolicy:
    return StepBoundaryPreemptiveElasticPolicy(
        topology=topology,
        task_runtime_estimator=task_runtime_estimator,
        mode="demote",
        reshard_ms=float(os.getenv("STEP_PREEMPT_RESHARD_MS", "0")),
        top_k=int(os.getenv("STEP_PREEMPT_TOP_K", "6")),
        demote_max_sp=int(os.getenv("STEP_PREEMPT_DEMOTE_MAX_SP", "2")),
        max_pause_ms=float(os.getenv("STEP_PREEMPT_MAX_PAUSE_MS", "120000")),
    )


def _request_class(request_id: str) -> str:
    prefix = request_id.split("_", 1)[0].upper()
    if prefix in {"S", "M", "L"}:
        return prefix
    return "M"


def _default_priority(request_class: str) -> int:
    return {"S": 100, "M": 50, "L": 0}.get(request_class, 50)


def _float_attr(value: object, name: str) -> float | None:
    raw = getattr(value, name, None)
    return float(raw) if raw is not None else None


def _rank_mask(ranks: tuple[int, ...]) -> int:
    mask = 0
    for rank in ranks:
        mask |= 1 << int(rank)
    return mask
