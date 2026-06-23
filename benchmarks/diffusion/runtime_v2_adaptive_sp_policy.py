#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Load-adaptive request-level SP policy for runtime_v2 policy simulation.

Use with runtime_v2_policy_simulator.py:

    --policy benchmarks.diffusion.runtime_v2_adaptive_sp_policy:make_policy

The workload stays placement-agnostic.  This policy chooses an execution group
at request admission time from the topology supplied to the simulator.
"""

from __future__ import annotations

import math
import os

from vllm_omni.diffusion.runtime_v2.protocol import (
    RequestExecutionPlan,
    TaskKind,
)
from vllm_omni.diffusion.runtime_v2.scheduler import (
    DynamicStepFCFSSchedulerPolicy,
    FCFSSchedulerPolicy,
    TaskRuntimeEstimator,
)
from vllm_omni.diffusion.runtime_v2.topology import RuntimeTopology


class LoadAdaptiveSPPolicy(FCFSSchedulerPolicy):
    """Bind each request to either latency-optimized or throughput-optimized SP.

    Low load:
        choose the group with the lowest estimated request latency.

    High load:
        choose the group with the lowest estimated GPU-time
        (request latency estimate * number of occupied ranks), then balance
        backlog across equivalent lanes.

    The policy is intentionally request-level.  It does not switch SP within a
    denoising loop, so RESHARD cost is not hidden inside the policy result.
    """

    def __init__(
        self,
        topology: RuntimeTopology,
        task_runtime_estimator: TaskRuntimeEstimator | None = None,
        *,
        high_load_interarrival_ms: float = 5_000.0,
        backlog_high_load_requests: int = 1,
        high_load_cooldown_ms: float = 30_000.0,
        latency_sp: int | None = None,
    ) -> None:
        super().__init__(
            topology=topology,
            task_runtime_estimator=task_runtime_estimator,
        )
        self.high_load_interarrival_ms = float(high_load_interarrival_ms)
        self.backlog_high_load_requests = int(backlog_high_load_requests)
        self.high_load_cooldown_ms = float(high_load_cooldown_ms)
        self.latency_sp = latency_sp
        self._last_arrival_ms: float | None = None
        self._high_load_until_ms = -math.inf

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
            high_load = self._is_high_load(plan)
            if self.latency_sp is not None and not high_load:
                latency_candidates = tuple(
                    group_id
                    for group_id in candidate_group_ids
                    if int(self.topology.get_group(group_id).parallel_spec.sp) == self.latency_sp
                )
                if latency_candidates:
                    selected_group_id = self._select_latency_group(plan, latency_candidates)
                else:
                    selected_group_id = self._select_latency_group(plan, candidate_group_ids)
                self._assign_plan_group(plan, selected_group_id)
                return selected_group_id
            selected_group_id = (
                self._select_throughput_group(plan, candidate_group_ids)
                if high_load
                else self._select_latency_group(plan, candidate_group_ids)
            )

        self._assign_plan_group(plan, selected_group_id)
        return selected_group_id

    def _assign_plan_group(self, plan: RequestExecutionPlan, selected_group_id: str) -> None:
        self.request_group[plan.request_id] = selected_group_id
        for task in plan.tasks.values():
            if task.group_id is None:
                task.group_id = selected_group_id
            elif task.group_id != selected_group_id:
                raise ValueError(
                    f"request {plan.request_id} contains mismatched task group_id={task.group_id!r}; "
                    f"expected {selected_group_id!r}"
                )

    def _is_high_load(self, plan: RequestExecutionPlan) -> bool:
        arrival_ms = _plan_arrival_ms(plan)
        if arrival_ms is not None:
            if self._last_arrival_ms is not None:
                interarrival_ms = arrival_ms - self._last_arrival_ms
                if interarrival_ms <= self.high_load_interarrival_ms:
                    self._high_load_until_ms = max(
                        self._high_load_until_ms,
                        arrival_ms + self.high_load_cooldown_ms,
                    )
            self._last_arrival_ms = arrival_ms
            if arrival_ms <= self._high_load_until_ms:
                return True
        return len(self.request_remaining_work) >= self.backlog_high_load_requests

    def _select_latency_group(
        self,
        plan: RequestExecutionPlan,
        candidate_group_ids: tuple[str, ...],
    ) -> str:
        return min(
            candidate_group_ids,
            key=lambda group_id: (
                self._estimate_plan_work(plan, self.topology.get_group(group_id).parallel_spec),
                self.group_backlog_work.get(group_id, 0.0),
                self._group_order.get(group_id, len(self._group_order)),
                group_id,
            ),
        )

    def _select_throughput_group(
        self,
        plan: RequestExecutionPlan,
        candidate_group_ids: tuple[str, ...],
    ) -> str:
        return min(
            candidate_group_ids,
            key=lambda group_id: self._throughput_score(plan, group_id),
        )

    def _throughput_score(
        self,
        plan: RequestExecutionPlan,
        group_id: str,
    ) -> tuple[float, float, int, str]:
        group = self.topology.get_group(group_id)
        estimated_ms = self._estimate_plan_work(plan, group.parallel_spec)
        rank_count = max(1, len(group.ranks))
        gpu_time_ms = estimated_ms * float(rank_count)
        return (
            gpu_time_ms,
            self.group_backlog_work.get(group_id, 0.0) + estimated_ms,
            self._group_order.get(group_id, len(self._group_order)),
            group_id,
        )


class StepLevelAdaptiveSPPolicy(DynamicStepFCFSSchedulerPolicy):
    """Policy-driven step-level SP switching for the simulator.

    A low-load request stays entirely on the latency group.  A high-load
    request starts on the latency group, runs middle denoise steps on one
    low-SP lane, then switches back for the final step so the existing plan
    boundary to VAE remains local.  The plan compiler inserts explicit RESHARD
    tasks at both step boundaries.
    """

    def __init__(
        self,
        topology: RuntimeTopology,
        task_runtime_estimator: TaskRuntimeEstimator | None = None,
        *,
        high_load_interarrival_ms: float = 5_000.0,
        high_load_cooldown_ms: float = 30_000.0,
    ) -> None:
        super().__init__(topology=topology)
        self.task_runtime_estimator = task_runtime_estimator
        self.high_load_interarrival_ms = float(high_load_interarrival_ms)
        self.high_load_cooldown_ms = float(high_load_cooldown_ms)
        self._last_arrival_ms: float | None = None
        self._high_load_until_ms = -math.inf
        self._lane_cursor = 0
        self._latency_group_id = self._select_latency_group_id()
        self._throughput_group_ids = self._select_throughput_group_ids()

    def build_sim_plan(self, *, request: object, default_builder: object) -> RequestExecutionPlan:
        builder = default_builder
        if not callable(builder):
            raise TypeError("default_builder must be callable")
        if not self._is_high_load_request(request) or int(getattr(request, "num_steps")) < 2:
            return builder(request, group_id=self._latency_group_id)

        compute_lane_group_id, aux_lane_group_id = self._next_throughput_pair()
        schedule = (
            {"start": 0, "end": 1, "group_id": self._latency_group_id},
            {"start": 1, "end": None, "group_id": compute_lane_group_id},
        )
        return builder(
            request,
            stage_group_ids={"aux": aux_lane_group_id, "dit": self._latency_group_id},
            dit_step_schedule=schedule,
        )

    def _is_high_load_request(self, request: object) -> bool:
        arrival_ms = float(getattr(request, "arrival_ms"))
        if self._last_arrival_ms is not None:
            interarrival_ms = arrival_ms - self._last_arrival_ms
            if interarrival_ms <= self.high_load_interarrival_ms:
                self._high_load_until_ms = max(
                    self._high_load_until_ms,
                    arrival_ms + self.high_load_cooldown_ms,
                )
        self._last_arrival_ms = arrival_ms
        if arrival_ms <= self._high_load_until_ms:
            return True
        return self._has_buffered_or_running_work()

    def _has_buffered_or_running_work(self) -> bool:
        if self.dispatched_kind:
            return True
        if self.group_ready:
            return True
        return bool(self.reshard_ready)

    def _select_latency_group_id(self) -> str:
        latency_sp = _latency_sp_from_env()
        if latency_sp is not None:
            candidates = [
                group
                for group in self.topology.groups
                if int(group.parallel_spec.sp) == latency_sp
            ]
            if candidates:
                return min(
                    candidates,
                    key=lambda group: (
                        self._group_order.get(group.group_id, len(self._group_order)),
                        group.group_id,
                    ),
                ).group_id
        return max(
            self.topology.groups,
            key=lambda group: (
                int(group.parallel_spec.sp),
                len(group.ranks),
                -self._group_order.get(group.group_id, 0),
            ),
        ).group_id

    def _select_throughput_group_ids(self) -> tuple[str, ...]:
        min_sp = min(int(group.parallel_spec.sp) for group in self.topology.groups)
        lanes = tuple(
            group.group_id
            for group in self.topology.groups
            if int(group.parallel_spec.sp) == min_sp
        )
        if not lanes:
            raise ValueError("step-level adaptive policy requires at least one throughput group")
        return lanes

    def _next_throughput_lane(self) -> str:
        group_id = self._throughput_group_ids[self._lane_cursor % len(self._throughput_group_ids)]
        self._lane_cursor += 1
        return group_id

    def _next_throughput_pair(self) -> tuple[str, str]:
        compute_index = self._lane_cursor % len(self._throughput_group_ids)
        compute_group_id = self._throughput_group_ids[compute_index]
        self._lane_cursor += 1
        if len(self._throughput_group_ids) < 2:
            return compute_group_id, self._latency_group_id
        aux_group_id = self._throughput_group_ids[(compute_index + 1) % len(self._throughput_group_ids)]
        return compute_group_id, aux_group_id


class OnlineStepLevelAdaptiveSPPolicy(DynamicStepFCFSSchedulerPolicy):
    """Choose the group for each runnable task from current online load."""

    def __init__(
        self,
        topology: RuntimeTopology,
        task_runtime_estimator: TaskRuntimeEstimator | None = None,
        *,
        high_load_interarrival_ms: float = 5_000.0,
    ) -> None:
        super().__init__(topology=topology)
        self.task_runtime_estimator = task_runtime_estimator
        self.high_load_interarrival_ms = float(high_load_interarrival_ms)
        self._last_arrival_ms: float | None = None
        self._high_load_active = False
        self._lane_cursor = 0
        self._request_lane: dict[str, str] = {}
        self._lane_assignment_counts: dict[str, int] = {}
        self._high_load_admitted_requests: set[str] = set()
        self._latency_group_id = self._select_latency_group_id()
        self._throughput_group_ids = self._select_throughput_group_ids(
            allowed_sps=_allowed_throughput_sps_from_env(),
            include_best_sp_only=True,
        )

    def on_request_submitted(self, plan: RequestExecutionPlan):
        arrival_ms = _plan_arrival_ms(plan)
        high_load_at_admission = self._has_other_work(plan.request_id) or self._high_load_active
        if arrival_ms is not None:
            if self._last_arrival_ms is not None:
                interarrival_ms = arrival_ms - self._last_arrival_ms
                if interarrival_ms <= self.high_load_interarrival_ms:
                    self._high_load_active = True
                    high_load_at_admission = True
            self._last_arrival_ms = arrival_ms
        if high_load_at_admission:
            self._high_load_admitted_requests.add(plan.request_id)
            self._lane_for_request(plan.request_id)
        return super().on_request_submitted(plan)

    def _select_group_for_task(self, task):
        high_load_now = self._is_high_load_now(task.request_id)
        if task.request_id in self._high_load_admitted_requests and high_load_now:
            return self._lane_for_request(task.request_id)
        if not high_load_now:
            return self._latency_group_id
        if task.kind == TaskKind.DIT_STEP_CHUNK:
            return self._lane_for_request(task.request_id)
        if task.kind in (TaskKind.TEXT_ENCODE, TaskKind.DIT_PREPARE, TaskKind.TIMESTEP_PREPARE):
            return self._latency_group_id
        return self._lane_for_request(task.request_id)

    def _is_high_load_now(self, request_id: str) -> bool:
        if self._has_other_work(request_id):
            return True
        return False

    def _has_other_work(self, request_id: str) -> bool:
        if any(_request_id_from_task_id(task_id) != request_id for task_id in self.dispatched_kind):
            return True
        if self.group_ready:
            for queue in self.group_ready.values():
                if any(task.request_id != request_id for task in queue):
                    return True
        return any(task.request_id != request_id for task in self.reshard_ready)

    def _lane_for_request(self, request_id: str) -> str:
        group_id = self._request_lane.get(request_id)
        if group_id is not None:
            return group_id
        group_id = min(
            self._throughput_group_ids,
            key=lambda candidate: (
                self._lane_assignment_counts.get(candidate, 0),
                self.outstanding_per_group.get(candidate, 0),
                len(self.group_ready.get(candidate, ())),
                self._group_order.get(candidate, len(self._group_order)),
                candidate,
            ),
        )
        self._request_lane[request_id] = group_id
        self._lane_assignment_counts[group_id] = self._lane_assignment_counts.get(group_id, 0) + 1
        return group_id

    def _drop_request(self, request_id: str) -> None:
        group_id = self._request_lane.pop(request_id, None)
        if group_id is not None:
            count = self._lane_assignment_counts.get(group_id, 0)
            if count <= 1:
                self._lane_assignment_counts.pop(group_id, None)
            else:
                self._lane_assignment_counts[group_id] = count - 1
        self._high_load_admitted_requests.discard(request_id)
        super()._drop_request(request_id)
        if not self.dispatched_kind and not self.group_ready and not self.reshard_ready:
            self._high_load_active = False

    def _select_latency_group_id(self) -> str:
        latency_sp = _latency_sp_from_env()
        if latency_sp is not None:
            candidates = [
                group
                for group in self.topology.groups
                if int(group.parallel_spec.sp) == latency_sp
            ]
            if candidates:
                return min(
                    candidates,
                    key=lambda group: (
                        self._group_order.get(group.group_id, len(self._group_order)),
                        group.group_id,
                    ),
                ).group_id
        return max(
            self.topology.groups,
            key=lambda group: (
                int(group.parallel_spec.sp),
                len(group.ranks),
                -self._group_order.get(group.group_id, 0),
            ),
        ).group_id

    def _select_throughput_group_ids(
        self,
        *,
        allowed_sps: set[int] | None = None,
        include_best_sp_only: bool = True,
    ) -> tuple[str, ...]:
        groups = [
            group
            for group in self.topology.groups
            if group.group_id != self._latency_group_id
        ]
        if not groups:
            groups = list(self.topology.groups)
        if allowed_sps:
            groups = [
                group
                for group in groups
                if int(group.parallel_spec.sp) in allowed_sps
            ]
        elif include_best_sp_only:
            min_sp = min(int(group.parallel_spec.sp) for group in groups)
            groups = [group for group in groups if int(group.parallel_spec.sp) == min_sp]
        if not groups:
            raise ValueError("no throughput groups remain after applying ADAPTIVE_SP_THROUGHPUT_SPS")
        return tuple(group.group_id for group in groups)


def make_policy(
    topology: RuntimeTopology,
    task_runtime_estimator: TaskRuntimeEstimator,
) -> LoadAdaptiveSPPolicy:
    return LoadAdaptiveSPPolicy(
        topology=topology,
        task_runtime_estimator=task_runtime_estimator,
        high_load_interarrival_ms=float(os.getenv("ADAPTIVE_SP_HIGH_LOAD_INTERARRIVAL_MS", "5000")),
        backlog_high_load_requests=int(os.getenv("ADAPTIVE_SP_BACKLOG_HIGH_LOAD_REQUESTS", "1")),
        high_load_cooldown_ms=float(os.getenv("ADAPTIVE_SP_HIGH_LOAD_COOLDOWN_MS", "30000")),
        latency_sp=_latency_sp_from_env(),
    )


def make_step_policy(
    topology: RuntimeTopology,
    task_runtime_estimator: TaskRuntimeEstimator,
) -> StepLevelAdaptiveSPPolicy:
    return StepLevelAdaptiveSPPolicy(
        topology=topology,
        task_runtime_estimator=task_runtime_estimator,
        high_load_interarrival_ms=float(os.getenv("ADAPTIVE_SP_HIGH_LOAD_INTERARRIVAL_MS", "5000")),
        high_load_cooldown_ms=float(os.getenv("ADAPTIVE_SP_HIGH_LOAD_COOLDOWN_MS", "30000")),
    )


def make_online_step_policy(
    topology: RuntimeTopology,
    task_runtime_estimator: TaskRuntimeEstimator,
) -> OnlineStepLevelAdaptiveSPPolicy:
    return OnlineStepLevelAdaptiveSPPolicy(
        topology=topology,
        task_runtime_estimator=task_runtime_estimator,
        high_load_interarrival_ms=float(os.getenv("ADAPTIVE_SP_HIGH_LOAD_INTERARRIVAL_MS", "5000")),
    )


def _resolve_explicit_request_group(plan: RequestExecutionPlan) -> str | None:
    explicit_groups = {task.group_id for task in plan.tasks.values() if task.group_id is not None}
    if not explicit_groups:
        return None
    if len(explicit_groups) != 1:
        raise ValueError(
            f"request {plan.request_id} contains tasks pinned to multiple groups: {sorted(explicit_groups)!r}"
        )
    return next(iter(explicit_groups))


def _candidate_groups_for_plan(
    topology: RuntimeTopology,
    plan: RequestExecutionPlan,
) -> tuple[str, ...]:
    return tuple(
        group.group_id
        for group in topology.groups
        if all(task.kind in group.supported_task_kinds for task in plan.tasks.values())
    )


def _plan_arrival_ms(plan: RequestExecutionPlan) -> float | None:
    value = plan.initial_artifacts[0].value if plan.initial_artifacts else None
    arrival = getattr(value, "arrival_ms", None)
    return float(arrival) if arrival is not None else None


def _request_id_from_task_id(task_id: str) -> str:
    return task_id.split(":", 1)[0]


def _allowed_throughput_sps_from_env() -> set[int] | None:
    raw = os.getenv("ADAPTIVE_SP_THROUGHPUT_SPS", "").strip()
    if not raw:
        return None
    return {int(part.strip()) for part in raw.split(",") if part.strip()}


def _latency_sp_from_env() -> int | None:
    raw = os.getenv("ADAPTIVE_SP_LATENCY_SP", "").strip()
    return int(raw) if raw else None
