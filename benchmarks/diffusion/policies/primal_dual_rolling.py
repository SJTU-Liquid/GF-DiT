# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import os
from collections import deque
from dataclasses import dataclass, field
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
    TaskRuntimeEstimator,
)
from vllm_omni.diffusion.runtime_v2.topology import RuntimeTopology


@dataclass(frozen=True)
class _RequestMeta:
    deadline_ms: float | None
    priority: int
    arrival_ms: float
    order: int
    request_class: str
    t_single_ms: float


@dataclass(frozen=True)
class _PlanAction:
    request_id: str
    task_id: str
    group_id: str
    ranks: tuple[int, ...]
    start_ms: float
    finish_ms: float
    duration_ms: float
    reshard_ms: float
    sp_size: int
    task_kind: TaskKind


@dataclass(frozen=True)
class _HoldAction:
    until_ms: float
    reason: str
    penalty: float


@dataclass
class _ShadowState:
    rank_free_at: tuple[float, ...]
    head_index: dict[str, int]
    ready_time: dict[str, float]
    last_group: dict[str, str | None]
    finish_times: dict[str, float] = field(default_factory=dict)
    actions: tuple[_PlanAction, ...] = ()
    hold: _HoldAction | None = None


class PrimalDualRollingPlannerPolicy(DynamicStepFCFSSchedulerPolicy):
    """Rolling-horizon simulator policy with observed SLO virtual queues.

    The policy keeps the simulator contract unchanged: returned tasks are only
    the immediately executable prefix.  Future completion times are produced by
    an internal shadow rollout over the currently active requests.
    """

    _SINGLE_RANK_TASK_KINDS = (
        TaskKind.TEXT_ENCODE,
        TaskKind.VAE_DECODE,
        TaskKind.FINALIZE,
    )
    _DIT_TASK_KINDS = (
        TaskKind.DIT_PREPARE,
        TaskKind.TIMESTEP_PREPARE,
        TaskKind.DIT_STEP_CHUNK,
    )
    _STANDARD_TASK_KINDS = _SINGLE_RANK_TASK_KINDS + _DIT_TASK_KINDS

    def __init__(
        self,
        topology: RuntimeTopology,
        task_runtime_estimator: TaskRuntimeEstimator | None = None,
        *,
        horizon_steps: int = 10,
        beam_size: int = 12,
        top_m_requests: int = 8,
        top_g_groups: int = 8,
        reshard_ms: float = 0.0,
        dual_step_size: float = 4.0,
        ewma_alpha: float = 0.20,
        target_miss_rate: dict[str, float] | None = None,
        initial_lambda: dict[str, float] | None = None,
        lambda_cap: float = 80.0,
        latency_weight: dict[str, float] | None = None,
        hold_penalty_weight: float = 0.02,
        reshard_penalty_weight: float = 0.0,
        gpu_time_penalty_weight: float = 0.2,
        pressure_threshold: float = 1.0,
        predictive_dual: bool = True,
        predictive_dual_throttle_ms: float = 500.0,
        allowed_sp_sizes: Iterable[int] | None = None,
        final_edf_tail_states: int = 0,
        batch_states_per_mask: int = 1,
        batch_final_tail_states: int = 0,
        planner_mode: str = "batch",
        debug: bool = False,
    ) -> None:
        super().__init__(topology=topology)
        self.task_runtime_estimator = task_runtime_estimator
        self.horizon_steps = max(1, int(horizon_steps))
        self.beam_size = max(1, int(beam_size))
        self.top_m_requests = max(1, int(top_m_requests))
        self.top_g_groups = max(1, int(top_g_groups))
        self.reshard_ms = max(0.0, float(reshard_ms))
        self.dual_step_size = max(0.0, float(dual_step_size))
        self.ewma_alpha = min(1.0, max(0.0, float(ewma_alpha)))
        self.target_miss_rate = {
            "S": 0.05,
            "M": 0.05,
            "L": 0.10,
            **(target_miss_rate or {}),
        }
        self.lambda_by_class = {
            "S": 0.0,
            "M": 0.0,
            "L": 0.0,
            **(initial_lambda or {}),
        }
        self.lambda_cap = max(0.0, float(lambda_cap))
        self.latency_weight = {
            "S": 1.0,
            "M": 1.0,
            "L": 1.0,
            **(latency_weight or {}),
        }
        self.hold_penalty_weight = max(0.0, float(hold_penalty_weight))
        self.reshard_penalty_weight = max(0.0, float(reshard_penalty_weight))
        self.gpu_time_penalty_weight = max(0.0, float(gpu_time_penalty_weight))
        self.pressure_threshold = max(0.0, float(pressure_threshold))
        self.predictive_dual = bool(predictive_dual)
        self.predictive_dual_throttle_ms = max(0.0, float(predictive_dual_throttle_ms))
        self._last_predictive_lambda_ms = -math.inf
        self.allowed_sp_sizes = self._normalize_allowed_sp_sizes(allowed_sp_sizes)
        self.final_edf_tail_states = max(0, int(final_edf_tail_states))
        self.batch_states_per_mask = max(1, int(batch_states_per_mask))
        self.batch_final_tail_states = max(0, int(batch_final_tail_states))
        if planner_mode not in {"beam", "batch"}:
            raise ValueError(f"planner_mode must be beam or batch, got {planner_mode!r}")
        self.planner_mode = planner_mode
        self.debug = bool(debug)

        self.ready_by_request: dict[str, deque[InferenceTask]] = {}
        self.request_meta: dict[str, _RequestMeta] = {}
        self.request_order: dict[str, int] = {}
        self.request_inflight: dict[str, int] = {}
        self.request_last_group: dict[str, str] = {}
        self.request_running_task: dict[str, str] = {}
        self.predicted_running: dict[str, tuple[float, tuple[int, ...], str, str]] = {}
        self.observed_miss_ewma: dict[str, float] = {}
        self._request_seq = 0
        self._task_index: dict[str, InferenceTask] = {}
        self._task_position: dict[str, tuple[str, int]] = {}
        self._request_task_sequence: dict[str, tuple[InferenceTask, ...]] = {}
        self._candidate_group_cache: dict[TaskKind, tuple[ExecutionGroupSpec, ...]] = {}
        self._duration_cache: dict[tuple[str, str], float] = {}
        self._best_remaining_cache: dict[tuple[str, int, str | None], float] = {}
        self.now_ms = 0.0

    # ------------------------------------------------------------------
    # Policy lifecycle
    # ------------------------------------------------------------------

    def build_sim_plan(self, *, request, default_builder):
        plan = default_builder(request)
        for task in plan.tasks.values():
            task.group_id = None
        return plan

    def on_request_submitted(self, plan: RequestExecutionPlan):
        self._request_seq += 1
        value = plan.initial_artifacts[0].value if plan.initial_artifacts else None
        meta = self._meta_from_value(value, order=self._request_seq, plan=plan)
        self.now_ms = max(self.now_ms, meta.arrival_ms)
        self.request_meta[plan.request_id] = meta
        self.request_order[plan.request_id] = self._request_seq
        sequence = self._linear_task_sequence(plan)
        self._request_task_sequence[plan.request_id] = sequence
        for idx, task in enumerate(sequence):
            self._task_index[task.task_id] = task
            self._task_position[task.task_id] = (plan.request_id, idx)
        root_tasks = [task for task in sequence if not task.dependencies]
        return self.on_tasks_runnable(root_tasks)

    def on_tasks_runnable(self, tasks: Iterable[InferenceTask]):
        saw_new_task = False
        for task in tasks:
            saw_new_task = True
            task.status = TaskStatus.READY
            self._task_index[task.task_id] = task
            self.ready_by_request.setdefault(task.request_id, deque()).append(task)
        if not saw_new_task:
            return []
        return self._take_dispatchable_tasks()

    def on_worker_event(self, event: WorkerEvent):
        self.now_ms = max(self.now_ms, float(event.timestamp_ns) / 1_000_000.0)
        if event.kind in (WorkerEventKind.TASK_EXEC_END, WorkerEventKind.TASK_FAILED):
            self._release_dispatched_task(event)
            return self._take_dispatchable_tasks()
        if event.kind in (WorkerEventKind.REQUEST_FINISHED, WorkerEventKind.REQUEST_FAILED):
            self._update_observed_dual(event)
            self._drop_request(event.request_id)
            return []
        return []

    # ------------------------------------------------------------------
    # Immediate commit from rolling-horizon shadow plan
    # ------------------------------------------------------------------

    def _take_dispatchable_tasks(self) -> list[InferenceTask]:
        if self.predictive_dual:
            self._refresh_predictive_lambdas()
        state = self._build_shadow_state()
        if not state.head_index and not self.predicted_running:
            return []
        if self.planner_mode == "batch":
            immediate = self._batch_dp_actions(state)
            if immediate:
                out: list[InferenceTask] = []
                for action in immediate:
                    self._dispatch_action(action, out)
                return out
            if self._safe_to_hold():
                return []
            fallback = self._fallback_greedy_action(state)
            if fallback is None:
                return []
            out: list[InferenceTask] = []
            self._dispatch_action(fallback, out)
            self._debug_plan(state, committed=(fallback,), fallback=True)
            return out
        plan = self._beam_rollout(state)
        immediate = self._extract_immediate_actions(plan)
        if immediate:
            out: list[InferenceTask] = []
            for action in immediate:
                self._dispatch_action(action, out)
            self._debug_plan(plan, committed=immediate)
            return out
        if self._safe_to_hold():
            self._debug_plan(plan, committed=())
        return []

    def _batch_dp_actions(self, state: _ShadowState) -> tuple[_PlanAction, ...]:
        candidates_by_request: list[tuple[str, list[tuple[float, _PlanAction]]]] = []
        for request_id in self._top_rollout_requests(state):
            if self.request_inflight.get(request_id, 0) > 0:
                continue
            task = self._state_head_task(state, request_id)
            if task is None:
                continue
            scored: list[tuple[float, _PlanAction]] = []
            for action in self._candidate_actions_for_task(state, request_id, task):
                if action.start_ms > self.now_ms + 1e-6:
                    continue
                if not self._can_dispatch_ranks(action.ranks):
                    continue
                scored.append((self._batch_action_heuristic(state, action), action))
            if scored:
                scored.sort(key=lambda item: (-item[0], item[1].finish_ms, -item[1].sp_size, item[1].group_id))
                candidates_by_request.append((request_id, scored[: self.top_g_groups]))

        dp: dict[int, list[tuple[float, tuple[_PlanAction, ...]]]] = {0: [(0.0, ())]}
        for _request_id, scored_actions in candidates_by_request:
            next_dp = {mask: list(entries) for mask, entries in dp.items()}
            for used_mask, entries in dp.items():
                for _value, actions in entries:
                    for _action_score, action in scored_actions:
                        mask = self._rank_mask(action.ranks)
                        if used_mask & mask:
                            continue
                        new_mask = used_mask | mask
                        new_actions = (*actions, action)
                        next_state = self._apply_parallel_shadow_actions(state, new_actions)
                        objective = self._objective_with_best_case_tail(next_state)
                        if not math.isfinite(objective):
                            continue
                        next_dp.setdefault(new_mask, []).append((-objective, new_actions))
            dp = self._prune_batch_states(next_dp)

        scored_plans: list[tuple[float, float, tuple[_PlanAction, ...], _ShadowState]] = []
        for entries in dp.values():
            for heuristic, actions in entries:
                if not actions:
                    continue
                next_state = self._apply_parallel_shadow_actions(state, actions)
                objective = self._objective_with_best_case_tail(next_state)
                if math.isfinite(objective):
                    scored_plans.append((objective, -heuristic, actions, next_state))
        if not scored_plans:
            return ()

        scored_plans.sort(key=lambda item: (item[0], item[1], len(item[2])))
        if self.batch_final_tail_states > 0:
            best = min(
                scored_plans[: self.batch_final_tail_states],
                key=lambda item: (self._objective_with_tail(item[3]), item[0], item[1], len(item[2])),
            )
        else:
            best = scored_plans[0]
        return tuple(sorted(best[2], key=lambda action: (action.start_ms, action.finish_ms, action.task_id)))

    def _prune_batch_states(
        self,
        states_by_mask: dict[int, list[tuple[float, tuple[_PlanAction, ...]]]],
    ) -> dict[int, list[tuple[float, tuple[_PlanAction, ...]]]]:
        pruned: dict[int, list[tuple[float, tuple[_PlanAction, ...]]]] = {}
        for mask, states in states_by_mask.items():
            states.sort(key=lambda item: (-item[0], len(item[1])))
            pruned[mask] = states[: self.batch_states_per_mask]
        return pruned

    def _apply_parallel_shadow_actions(
        self,
        state: _ShadowState,
        actions: tuple[_PlanAction, ...],
    ) -> _ShadowState:
        next_state = state
        for action in sorted(actions, key=lambda item: (item.start_ms, item.finish_ms, item.task_id)):
            next_state = self._apply_shadow_action(next_state, action)
        return next_state

    def _batch_action_heuristic(self, state: _ShadowState, action: _PlanAction) -> float:
        meta = self.request_meta[action.request_id]
        t_single = self._t_single(meta)
        head_idx = state.head_index.get(action.request_id, 0)
        predicted_c = action.finish_ms + self._best_case_remaining_ms(
            action.request_id,
            head_idx + 1,
            action.group_id,
        )
        latency_norm = max(0.0, predicted_c - meta.arrival_ms) / t_single
        pressure = 0.0
        slack_norm = math.inf
        if meta.deadline_ms is not None:
            slack_norm = (meta.deadline_ms - predicted_c) / t_single
            # Offset hinge: lambda active whenever slack_norm < pressure_threshold,
            # not only after the deadline is crossed. Lets the dual penalty grow
            # smoothly as slack shrinks instead of staying at 0 until miss.
            pressure = max(0.0, self.pressure_threshold - slack_norm)
        lam = self.lambda_by_class.get(meta.request_class, 0.0)
        priority = max(0.0, float(meta.priority)) / 100.0
        gpu_time_norm = (
            action.duration_ms * max(1, len(action.ranks))
        ) / (t_single * max(1, len(self.topology.workers)))
        deadline_pressure = 0.0 if slack_norm == math.inf else 1.0 / (1.0 + max(0.0, slack_norm))
        return (
            - self.latency_weight.get(meta.request_class, 1.0) * latency_norm
            - lam * pressure
            - self.gpu_time_penalty_weight * gpu_time_norm
            + 0.25 * deadline_pressure
            + 0.05 * priority
        )

    def _extract_immediate_actions(self, plan: _ShadowState) -> tuple[_PlanAction, ...]:
        out: list[_PlanAction] = []
        claimed: set[int] = set()
        seen_requests: set[str] = set()
        for action in sorted(plan.actions, key=lambda item: (item.start_ms, item.finish_ms, item.task_id)):
            if action.start_ms > self.now_ms + 1e-6:
                continue
            if action.request_id in seen_requests:
                continue
            if any(rank in claimed for rank in action.ranks):
                continue
            queue = self.ready_by_request.get(action.request_id)
            if not queue or queue[0].task_id != action.task_id:
                continue
            if self.request_inflight.get(action.request_id, 0) > 0:
                continue
            if not self._can_dispatch_ranks(action.ranks):
                continue
            out.append(action)
            seen_requests.add(action.request_id)
            claimed.update(action.ranks)
        return tuple(out)

    def _dispatch_action(self, action: _PlanAction, out: list[InferenceTask]) -> None:
        queue = self.ready_by_request.get(action.request_id)
        if not queue or queue[0].task_id != action.task_id:
            return
        task = queue.popleft()
        if not queue:
            self.ready_by_request.pop(action.request_id, None)
        task.group_id = action.group_id
        task.status = TaskStatus.DISPATCHED
        self._claim_ranks(action.ranks, reshard=False)
        self.outstanding_per_group[action.group_id] = self.outstanding_per_group.get(action.group_id, 0) + 1
        self.dispatched_kind[task.task_id] = action.group_id
        self.dispatched_ranks_by_task[task.task_id] = action.ranks
        self.request_inflight[action.request_id] = self.request_inflight.get(action.request_id, 0) + 1
        self.request_running_task[action.request_id] = task.task_id
        self.request_last_group[action.request_id] = action.group_id
        self.predicted_running[task.task_id] = (
            action.finish_ms,
            action.ranks,
            action.group_id,
            action.request_id,
        )
        out.append(task)

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
        if self.request_running_task.get(request_id) == event.task_id:
            self.request_running_task.pop(request_id, None)
        self.predicted_running.pop(event.task_id, None)
        if event.group_id:
            self.request_last_group[request_id] = event.group_id

    def _drop_request(self, request_id: str) -> None:
        self.ready_by_request.pop(request_id, None)
        self.request_meta.pop(request_id, None)
        self.request_order.pop(request_id, None)
        self.request_inflight.pop(request_id, None)
        self.request_last_group.pop(request_id, None)
        self.request_running_task.pop(request_id, None)
        self._request_task_sequence.pop(request_id, None)

    # ------------------------------------------------------------------
    # Shadow state and rollout
    # ------------------------------------------------------------------

    def _build_shadow_state(self) -> _ShadowState:
        rank_count = max((worker.worker_rank for worker in self.topology.workers), default=-1) + 1
        rank_free = [self.now_ms for _ in range(rank_count)]
        predicted_finish_by_request: dict[str, float] = {}
        for _task_id, (finish_ms, ranks, _group_id, request_id) in self.predicted_running.items():
            finish_ms = max(self.now_ms, float(finish_ms))
            predicted_finish_by_request[request_id] = max(
                predicted_finish_by_request.get(request_id, self.now_ms),
                finish_ms,
            )
            for rank in ranks:
                if 0 <= rank < len(rank_free):
                    rank_free[rank] = max(rank_free[rank], finish_ms)

        head_index: dict[str, int] = {}
        ready_time: dict[str, float] = {}
        last_group: dict[str, str | None] = {
            request_id: group_id for request_id, group_id in self.request_last_group.items()
        }
        finish_times: dict[str, float] = {}
        for request_id in self.request_meta:
            queue = self.ready_by_request.get(request_id)
            if queue:
                pos = self._task_position.get(queue[0].task_id)
                if pos is not None:
                    head_index[request_id] = pos[1]
                    ready_time[request_id] = (
                        predicted_finish_by_request.get(request_id, self.now_ms)
                        if self.request_inflight.get(request_id, 0) > 0
                        else self.now_ms
                    )
                continue
            running_task_id = self.request_running_task.get(request_id)
            if running_task_id is not None:
                pos = self._task_position.get(running_task_id)
                finish_ms = predicted_finish_by_request.get(request_id, self.now_ms)
                if pos is None:
                    continue
                next_idx = pos[1] + 1
                sequence = self._request_task_sequence.get(request_id, ())
                if next_idx < len(sequence):
                    head_index[request_id] = next_idx
                    ready_time[request_id] = finish_ms
                else:
                    finish_times[request_id] = finish_ms
        return _ShadowState(
            rank_free_at=tuple(rank_free),
            head_index=head_index,
            ready_time=ready_time,
            last_group=last_group,
            finish_times=finish_times,
        )

    def _beam_rollout(self, initial: _ShadowState) -> _ShadowState:
        beam: list[tuple[float, _ShadowState]] = [(self._objective_with_best_case_tail(initial), initial)]
        best_state = initial
        for _depth in range(self.horizon_steps):
            expanded: list[tuple[float, _ShadowState]] = []
            for _score, state in beam:
                candidates = self._enumerate_state_actions(state)
                if not candidates:
                    expanded.append((self._objective_with_best_case_tail(state), state))
                    continue
                for action in candidates:
                    next_state = self._apply_shadow_action(state, action)
                    score = self._objective_with_best_case_tail(next_state)
                    expanded.append((score, next_state))
            if not expanded:
                break
            expanded.sort(key=lambda item: item[0])
            beam = expanded[: self.beam_size]
            # The initial state's objective already includes a stateless tail
            # estimate, so explicit actions can tie it.  Prefer the best
            # expanded state so the policy commits the rollout's immediate
            # prefix instead of silently degenerating to fallback greedy.
            best_state = beam[0][1]
            if not any(state.head_index for _score, state in beam):
                break
        # Run the more faithful stateless EDF tail only on the final beam,
        # instead of every expansion.  This keeps online planning bounded while
        # preserving the intended tail model for final action selection.
        if beam and self.final_edf_tail_states > 0:
            tail_candidates = [state for _score, state in beam[: self.final_edf_tail_states]]
            best_state = min(tail_candidates, key=self._objective_with_tail)
        return self._with_hold_annotation(best_state)

    def _enumerate_state_actions(self, state: _ShadowState) -> list[_PlanAction]:
        request_ids = self._top_rollout_requests(state)
        actions: list[_PlanAction] = []
        for request_id in request_ids:
            task = self._state_head_task(state, request_id)
            if task is None:
                continue
            candidates = self._candidate_actions_for_task(state, request_id, task)
            actions.extend(candidates[: self.top_g_groups])
        actions.sort(key=lambda item: (item.start_ms, item.finish_ms, -item.sp_size, item.request_id))
        return actions

    def _apply_shadow_action(self, state: _ShadowState, action: _PlanAction) -> _ShadowState:
        rank_free = list(state.rank_free_at)
        for rank in action.ranks:
            rank_free[rank] = max(rank_free[rank], action.finish_ms)
        head_index = dict(state.head_index)
        ready_time = dict(state.ready_time)
        last_group = dict(state.last_group)
        finish_times = dict(state.finish_times)

        sequence = self._request_task_sequence.get(action.request_id, ())
        next_idx = head_index[action.request_id] + 1
        last_group[action.request_id] = action.group_id
        if next_idx < len(sequence):
            head_index[action.request_id] = next_idx
            ready_time[action.request_id] = action.finish_ms
        else:
            head_index.pop(action.request_id, None)
            ready_time.pop(action.request_id, None)
            finish_times[action.request_id] = action.finish_ms
        return _ShadowState(
            rank_free_at=tuple(rank_free),
            head_index=head_index,
            ready_time=ready_time,
            last_group=last_group,
            finish_times=finish_times,
            actions=(*state.actions, action),
        )

    def _with_hold_annotation(self, state: _ShadowState) -> _ShadowState:
        if state.actions and state.actions[0].start_ms > self.now_ms + 1e-6:
            until = state.actions[0].start_ms
            t_ref = self._reference_t_single_ms()
            penalty = self.hold_penalty_weight * max(0.0, until - self.now_ms) / t_ref
            return _ShadowState(
                rank_free_at=state.rank_free_at,
                head_index=state.head_index,
                ready_time=state.ready_time,
                last_group=state.last_group,
                finish_times=state.finish_times,
                actions=state.actions,
                hold=_HoldAction(until_ms=until, reason="wait_for_predicted_release", penalty=penalty),
            )
        return state

    def _top_rollout_requests(self, state: _ShadowState) -> list[str]:
        items = [request_id for request_id in state.head_index if request_id in self.request_meta]
        items.sort(key=lambda request_id: self._rollout_priority_key(state, request_id))
        return items[: self.top_m_requests]

    def _rollout_priority_key(self, state: _ShadowState, request_id: str) -> tuple[float, float, float, int, str]:
        meta = self.request_meta[request_id]
        task = self._state_head_task(state, request_id)
        if task is None:
            return (math.inf, math.inf, math.inf, meta.order, request_id)
        min_remaining = self._best_case_remaining_ms(request_id, state.head_index[request_id], state.last_group.get(request_id))
        deadline = meta.deadline_ms if meta.deadline_ms is not None else math.inf
        t_single = self._t_single(meta)
        miss_norm = 0.0 if deadline == math.inf else max(0.0, state.ready_time[request_id] + min_remaining - deadline) / t_single
        lam = self.lambda_by_class.get(meta.request_class, 0.0)
        slack_norm = (deadline - state.ready_time[request_id]) / t_single if deadline != math.inf else math.inf
        return (-lam * miss_norm, slack_norm, -meta.priority, meta.order, request_id)

    def _state_head_task(self, state: _ShadowState, request_id: str) -> InferenceTask | None:
        idx = state.head_index.get(request_id)
        if idx is None:
            return None
        sequence = self._request_task_sequence.get(request_id, ())
        if 0 <= idx < len(sequence):
            return sequence[idx]
        return None

    # ------------------------------------------------------------------
    # Objective and stateless tail evaluator
    # ------------------------------------------------------------------

    def _objective_with_tail(self, state: _ShadowState) -> float:
        completion = self._estimate_tail_completion_times(state)
        return self._objective_from_completion(state, completion)

    def _objective_with_best_case_tail(self, state: _ShadowState) -> float:
        completion = self._estimate_best_case_completion_times(state)
        return self._objective_from_completion(state, completion)

    def _objective_from_completion(self, state: _ShadowState, completion: dict[str, float]) -> float:
        if not completion and not self.request_meta:
            return 0.0
        score = 0.0
        for request_id, meta in self.request_meta.items():
            finish_ms = completion.get(request_id)
            if finish_ms is None or not math.isfinite(finish_ms):
                return math.inf
            t_single = self._t_single(meta)
            latency_norm = max(0.0, finish_ms - meta.arrival_ms) / t_single
            score += self.latency_weight.get(meta.request_class, 1.0) * latency_norm
            if meta.deadline_ms is not None:
                slack_norm = (meta.deadline_ms - finish_ms) / t_single
                pressure = max(0.0, self.pressure_threshold - slack_norm)
                score += self.lambda_by_class.get(meta.request_class, 0.0) * pressure
        if state.actions:
            t_ref = self._reference_t_single_ms()
            reshard_norm = sum(action.reshard_ms for action in state.actions) / t_ref
            score += self.reshard_penalty_weight * reshard_norm
            gpu_time_norm = sum(
                action.duration_ms * max(1, len(action.ranks))
                for action in state.actions
            ) / (t_ref * max(1, len(self.topology.workers)))
            score += self.gpu_time_penalty_weight * gpu_time_norm
            first_start = state.actions[0].start_ms
            if first_start > self.now_ms + 1e-6:
                score += self.hold_penalty_weight * (first_start - self.now_ms) / t_ref
        return score

    def _estimate_best_case_completion_times(self, state: _ShadowState) -> dict[str, float]:
        finish_times = dict(state.finish_times)
        for request_id, meta in self.request_meta.items():
            if request_id in finish_times:
                continue
            head_idx = state.head_index.get(request_id)
            if head_idx is None:
                finish_times[request_id] = self.now_ms
                continue
            task = self._sequence_task(request_id, head_idx)
            if task is None:
                finish_times[request_id] = self.now_ms
                continue
            first_actions = self._candidate_actions_for_task_raw(
                task=task,
                request_id=request_id,
                rank_free_at=state.rank_free_at,
                ready_ms=state.ready_time.get(request_id, self.now_ms),
                source_group_id=state.last_group.get(request_id),
                top_k=1,
            )
            if not first_actions:
                finish_times[request_id] = math.inf
                continue
            first = first_actions[0]
            finish_times[request_id] = first.finish_ms + self._best_case_remaining_ms(
                request_id,
                head_idx + 1,
                first.group_id,
            )
        return finish_times

    def _estimate_tail_completion_times(self, state: _ShadowState) -> dict[str, float]:
        rank_free = list(state.rank_free_at)
        head_index = dict(state.head_index)
        ready_time = dict(state.ready_time)
        last_group = dict(state.last_group)
        finish_times = dict(state.finish_times)
        max_steps = sum(
            max(0, len(self._request_task_sequence.get(request_id, ())) - idx)
            for request_id, idx in head_index.items()
        )
        for _ in range(max_steps):
            unfinished = [request_id for request_id in head_index if request_id in self.request_meta]
            if not unfinished:
                break
            unfinished.sort(key=lambda request_id: self._edf_tail_key(ready_time, request_id))
            selected: _PlanAction | None = None
            for request_id in unfinished:
                task = self._sequence_task(request_id, head_index[request_id])
                if task is None:
                    continue
                candidates = self._candidate_actions_for_task_raw(
                    task=task,
                    request_id=request_id,
                    rank_free_at=tuple(rank_free),
                    ready_ms=ready_time.get(request_id, self.now_ms),
                    source_group_id=last_group.get(request_id),
                    top_k=max(self.top_g_groups, 1),
                )
                if candidates:
                    selected = candidates[0]
                    break
            if selected is None:
                for request_id in unfinished:
                    finish_times.setdefault(request_id, math.inf)
                break
            for rank in selected.ranks:
                rank_free[rank] = max(rank_free[rank], selected.finish_ms)
            next_idx = head_index[selected.request_id] + 1
            sequence = self._request_task_sequence.get(selected.request_id, ())
            last_group[selected.request_id] = selected.group_id
            if next_idx < len(sequence):
                head_index[selected.request_id] = next_idx
                ready_time[selected.request_id] = selected.finish_ms
            else:
                head_index.pop(selected.request_id, None)
                ready_time.pop(selected.request_id, None)
                finish_times[selected.request_id] = selected.finish_ms
        for request_id in self.request_meta:
            if request_id not in finish_times and request_id not in head_index:
                finish_times[request_id] = self.now_ms
        return finish_times

    def _edf_tail_key(self, ready_time: dict[str, float], request_id: str) -> tuple[float, int, float, int, str]:
        meta = self.request_meta[request_id]
        deadline = meta.deadline_ms if meta.deadline_ms is not None else math.inf
        return (deadline, -meta.priority, ready_time.get(request_id, self.now_ms), meta.order, request_id)

    # ------------------------------------------------------------------
    # Candidate generation and cost estimation
    # ------------------------------------------------------------------

    def _candidate_actions_for_task(
        self,
        state: _ShadowState,
        request_id: str,
        task: InferenceTask,
    ) -> list[_PlanAction]:
        return self._candidate_actions_for_task_raw(
            task=task,
            request_id=request_id,
            rank_free_at=state.rank_free_at,
            ready_ms=state.ready_time.get(request_id, self.now_ms),
            source_group_id=state.last_group.get(request_id),
            top_k=self.top_g_groups,
        )

    def _candidate_actions_for_task_raw(
        self,
        *,
        task: InferenceTask,
        request_id: str,
        rank_free_at: tuple[float, ...],
        ready_ms: float,
        source_group_id: str | None,
        top_k: int,
    ) -> list[_PlanAction]:
        out: list[_PlanAction] = []
        groups = self._candidate_groups_for_state(
            task=task,
            rank_free_at=rank_free_at,
            ready_ms=ready_ms,
        )
        for group in groups:
            ranks = self._task_ranks_for_group(task, group)
            if not ranks:
                continue
            start_ms = max(ready_ms, max(rank_free_at[rank] for rank in ranks))
            reshard_cost_ms = self._reshard_cost_for(task, source_group_id, group.group_id)
            try:
                duration_ms = self._estimate_task_ms(task, group.group_id) + reshard_cost_ms
            except Exception:
                continue
            finish_ms = start_ms + duration_ms
            out.append(
                _PlanAction(
                    request_id=request_id,
                    task_id=task.task_id,
                    group_id=group.group_id,
                    ranks=ranks,
                    start_ms=start_ms,
                    finish_ms=finish_ms,
                    duration_ms=duration_ms,
                    reshard_ms=reshard_cost_ms,
                    sp_size=len(group.ranks),
                    task_kind=task.kind,
                )
            )
        out.sort(key=lambda action: (action.finish_ms, action.start_ms, action.reshard_ms, -action.sp_size, action.group_id))
        return self._select_diverse_sp_actions(out, top_k)

    def _candidate_groups_for_state(
        self,
        *,
        task: InferenceTask,
        rank_free_at: tuple[float, ...],
        ready_ms: float,
    ) -> tuple[ExecutionGroupSpec, ...]:
        groups: list[ExecutionGroupSpec] = list(self._candidate_groups(task))
        if task.kind not in self._DIT_TASK_KINDS:
            return tuple(groups)

        seen = {group.group_id for group in groups}

        def add_ranks(ranks: tuple[int, ...]) -> None:
            if not ranks:
                return
            sp_size = len(ranks)
            if sp_size not in self.allowed_sp_sizes:
                return
            group = self._ensure_group_for_ranks(
                ranks=tuple(sorted(int(rank) for rank in ranks)),
                supported_task_kinds=self._DIT_TASK_KINDS,
            )
            if group.group_id in seen:
                return
            seen.add(group.group_id)
            groups.append(group)

        ranks_by_free_time = tuple(
            rank for rank, _free_at in sorted(
                enumerate(rank_free_at),
                key=lambda item: (item[1], item[0]),
            )
        )
        free_now = tuple(
            rank for rank, free_at in enumerate(rank_free_at)
            if free_at <= ready_ms + 1e-6
        )
        for sp_size in self.allowed_sp_sizes:
            if sp_size <= 1 or sp_size > len(rank_free_at):
                continue
            if len(free_now) >= sp_size:
                add_ranks(tuple(sorted(free_now)[:sp_size]))
            add_ranks(tuple(ranks_by_free_time[:sp_size]))
        return tuple(groups)

    @staticmethod
    def _select_diverse_sp_actions(actions: list[_PlanAction], top_k: int) -> list[_PlanAction]:
        if len(actions) <= top_k:
            return actions
        selected: list[_PlanAction] = []
        seen_keys: set[tuple[str, str]] = set()
        by_sp: dict[int, list[_PlanAction]] = {}
        for action in actions:
            by_sp.setdefault(action.sp_size, []).append(action)

        # Keep a couple of rank-disjoint alternatives per SP.  This gives the
        # rank-DP enough choices to fill both halves of an 8-GPU cluster without
        # retaining every overlapping sliding window.
        for sp_size in sorted(by_sp, reverse=True):
            masks: list[int] = []
            for action in by_sp[sp_size]:
                mask = PrimalDualRollingPlannerPolicy._rank_mask(action.ranks)
                if any(mask & existing for existing in masks):
                    continue
                selected.append(action)
                masks.append(mask)
                seen_keys.add((action.task_id, action.group_id))
                if len(masks) >= 2 or len(selected) >= top_k:
                    break
            if len(selected) >= top_k:
                break

        for action in actions:
            key = (action.task_id, action.group_id)
            if key in seen_keys:
                continue
            selected.append(action)
            seen_keys.add((action.task_id, action.group_id))
            if len(selected) >= top_k:
                break
        selected.sort(key=lambda action: (action.finish_ms, action.start_ms, action.reshard_ms, -action.sp_size, action.group_id))
        return selected

    def _candidate_groups(self, task: InferenceTask) -> tuple[ExecutionGroupSpec, ...]:
        cached = self._candidate_group_cache.get(task.kind)
        if cached is not None:
            return cached
        groups: list[ExecutionGroupSpec] = []
        seen: set[str] = set()

        def add(group: ExecutionGroupSpec) -> None:
            if group.group_id in seen:
                return
            if task.kind not in group.supported_task_kinds:
                return
            if int(group.parallel_spec.tp) != 1 or int(group.parallel_spec.cfg) != 1:
                return
            sp = int(group.parallel_spec.sp)
            if sp not in self.allowed_sp_sizes:
                return
            seen.add(group.group_id)
            groups.append(group)

        if task.kind in self._SINGLE_RANK_TASK_KINDS:
            for rank in self._worker_ranks():
                add(self._ensure_group_for_ranks(ranks=(rank,), supported_task_kinds=self._STANDARD_TASK_KINDS))
            result = tuple(groups)
            self._candidate_group_cache[task.kind] = result
            return result

        for sp in self.allowed_sp_sizes:
            if sp < 1:
                continue
            for ranks in self._partition_rank_windows(sp):
                add(self._ensure_group_for_ranks(ranks=ranks, supported_task_kinds=self._DIT_TASK_KINDS))
        for group in self.topology.groups:
            add(group)
        for sp in self.allowed_sp_sizes:
            if sp <= 1:
                continue
            for ranks in self._sliding_rank_windows(sp):
                add(self._ensure_group_for_ranks(ranks=ranks, supported_task_kinds=self._DIT_TASK_KINDS))
        result = tuple(groups)
        self._candidate_group_cache[task.kind] = result
        return result

    def _sliding_rank_windows(self, sp_size: int) -> tuple[tuple[int, ...], ...]:
        ranks = self._worker_ranks()
        if sp_size > len(ranks):
            return ()
        if sp_size == len(ranks):
            return (ranks,)
        return tuple(ranks[idx : idx + sp_size] for idx in range(0, len(ranks) - sp_size + 1))

    def _partition_rank_windows(self, sp_size: int) -> tuple[tuple[int, ...], ...]:
        ranks = self._worker_ranks()
        if sp_size > len(ranks):
            return ()
        out: list[tuple[int, ...]] = []
        for idx in range(0, len(ranks), sp_size):
            window = ranks[idx : idx + sp_size]
            if len(window) == sp_size:
                out.append(window)
        return tuple(out)

    def _worker_ranks(self) -> tuple[int, ...]:
        return tuple(sorted(int(worker.worker_rank) for worker in self.topology.workers))

    def _task_ranks_for_group(self, task: InferenceTask, group: ExecutionGroupSpec) -> tuple[int, ...]:
        if task.kind in self._SINGLE_RANK_TASK_KINDS:
            return (int(group.ranks[0]),)
        return tuple(int(rank) for rank in group.ranks)

    def _reshard_cost_for(self, task: InferenceTask, source_group_id: str | None, target_group_id: str) -> float:
        if self.reshard_ms <= 0.0 or source_group_id is None:
            return 0.0
        if task.dependencies and source_group_id != target_group_id:
            return self.reshard_ms
        return 0.0

    def _estimate_task_ms(self, task: InferenceTask, group_id: str) -> float:
        key = (task.task_id, group_id)
        cached = self._duration_cache.get(key)
        if cached is not None:
            return cached
        if self.task_runtime_estimator is None:
            return 0.0
        group = self.topology.get_group(group_id)
        stage = str(task.payload.get("cost_model_stage") or task.kind.value)
        features = dict(task.payload.get("sim_features", {}))
        seq_len = int(task.payload.get("seq_len", features.get("latent_seq_len", 1)) or 1)
        estimated = float(self.task_runtime_estimator(group.parallel_spec, seq_len, stage))
        if task.kind == TaskKind.DIT_STEP_CHUNK:
            chunk_steps = int(features.get("chunk_steps", 1) or 1)
            estimated *= max(1, chunk_steps)
        estimated = max(0.0, estimated)
        self._duration_cache[key] = estimated
        return estimated

    def _best_case_remaining_ms(
        self,
        request_id: str,
        head_index: int,
        source_group_id: str | None,
    ) -> float:
        cache_key = (request_id, head_index, source_group_id)
        cached = self._best_remaining_cache.get(cache_key)
        if cached is not None:
            return cached
        total = 0.0
        last_group = source_group_id
        for task in self._request_task_sequence.get(request_id, ())[head_index:]:
            best: tuple[float, str] | None = None
            for group in self._candidate_groups(task):
                try:
                    duration_ms = self._estimate_task_ms(task, group.group_id)
                except Exception:
                    continue
                duration_ms += self._reshard_cost_for(task, last_group, group.group_id)
                if best is None or duration_ms < best[0]:
                    best = (duration_ms, group.group_id)
            if best is None:
                return math.inf
            total += best[0]
            last_group = best[1]
        self._best_remaining_cache[cache_key] = total
        return total

    def _sequence_task(self, request_id: str, index: int) -> InferenceTask | None:
        sequence = self._request_task_sequence.get(request_id, ())
        if 0 <= index < len(sequence):
            return sequence[index]
        return None

    def _fallback_greedy_action(self, state: _ShadowState) -> _PlanAction | None:
        ready = [request_id for request_id in self._top_rollout_requests(state) if self.request_inflight.get(request_id, 0) <= 0]
        for request_id in ready:
            task = self._state_head_task(state, request_id)
            if task is None:
                continue
            candidates = self._candidate_actions_for_task(state, request_id, task)
            for candidate in candidates:
                if candidate.start_ms <= self.now_ms + 1e-6 and self._can_dispatch_ranks(candidate.ranks):
                    return candidate
        return None

    # ------------------------------------------------------------------
    # Dual update and metadata
    # ------------------------------------------------------------------

    def _update_observed_dual(self, event: WorkerEvent) -> None:
        meta = self.request_meta.get(event.request_id)
        if meta is None or meta.deadline_ms is None or event.kind != WorkerEventKind.REQUEST_FINISHED:
            return
        miss = 1.0 if self.now_ms > meta.deadline_ms else 0.0
        old = self.observed_miss_ewma.get(meta.request_class, self.target_miss_rate.get(meta.request_class, 0.05))
        ewma = (1.0 - self.ewma_alpha) * old + self.ewma_alpha * miss
        self.observed_miss_ewma[meta.request_class] = ewma
        target = self.target_miss_rate.get(meta.request_class, 0.05)
        lam = self.lambda_by_class.get(meta.request_class, 0.0)
        lam = max(0.0, min(self.lambda_cap, lam + self.dual_step_size * (ewma - target)))
        self.lambda_by_class[meta.request_class] = lam
        if self.debug:
            print(
                "[pd-roll] dual_update "
                f"t={self.now_ms:.3f} request={event.request_id} class={meta.request_class} "
                f"miss={miss:.0f} ewma={ewma:.3f} target={target:.3f} lambda={lam:.3f}",
                flush=True,
            )

    def _refresh_predictive_lambdas(self) -> None:
        """Forward-looking dual update.

        Scans active requests, projects each one's earliest-possible completion
        (in-flight remainder + best-case unscheduled tasks), and counts
        predicted misses per class. Then does a Lyapunov-drift / dual-ascent step

            λ_c ← clip[λ_c + η (ŝ_c − τ_c), 0, λ_cap]

        where ŝ_c is the predicted miss-rate for class c right now and τ_c is
        the target miss-rate. Unlike ``_update_observed_dual`` (which only fires
        on REQUEST_FINISHED and lags far behind overload), this fires at every
        dispatch boundary so λ tracks current pressure.
        """
        if not self.request_meta:
            return
        now = self.now_ms
        if now - self._last_predictive_lambda_ms < self.predictive_dual_throttle_ms:
            return
        self._last_predictive_lambda_ms = now

        # Use a shadow rollout (EDF-tail estimator) so contention between
        # ready/inflight requests is reflected in the predicted finish times.
        # _best_case_remaining_ms would assume each request gets its optimal
        # SP exclusively, which produces predicted_miss ≈ 0 even under heavy
        # overload and starves the dual update of signal.
        shadow = self._build_shadow_state()
        completion = self._estimate_tail_completion_times(shadow)
        counts: dict[str, int] = {}
        # Continuous miss amount in t_single units (not binary 0/1). Binary
        # collapses to 1.0 once anyone overshoots, saturating EWMA and pinning
        # λ at the cap; continuous gives proportional signal so λ converges
        # near the SLO target instead of overshooting.
        miss_sum: dict[str, float] = {}
        for request_id, meta in self.request_meta.items():
            if meta.deadline_ms is None:
                continue
            predicted_c = completion.get(request_id)
            if predicted_c is None or not math.isfinite(predicted_c):
                continue
            cls = meta.request_class
            t_single = self._t_single(meta)
            miss_amount = max(0.0, (predicted_c - meta.deadline_ms) / t_single)
            counts[cls] = counts.get(cls, 0) + 1
            miss_sum[cls] = miss_sum.get(cls, 0.0) + miss_amount

        for cls, n in counts.items():
            avg_miss = miss_sum.get(cls, 0.0) / max(1, n)
            old = self.observed_miss_ewma.get(
                cls, self.target_miss_rate.get(cls, 0.05)
            )
            ewma = (1.0 - self.ewma_alpha) * old + self.ewma_alpha * avg_miss
            self.observed_miss_ewma[cls] = ewma
            target = self.target_miss_rate.get(cls, 0.05)
            lam = self.lambda_by_class.get(cls, 0.0)
            lam = max(
                0.0,
                min(self.lambda_cap, lam + self.dual_step_size * (ewma - target)),
            )
            self.lambda_by_class[cls] = lam
            if self.debug:
                print(
                    "[pd-roll] predictive_dual "
                    f"t={now:.3f} class={cls} n={n} avg_miss={avg_miss:.3f} "
                    f"ewma={ewma:.3f} target={target:.3f} lambda={lam:.3f}",
                    flush=True,
                )

    def _meta_from_value(self, value, *, order: int, plan: RequestExecutionPlan) -> _RequestMeta:
        raw_class = getattr(value, "request_class", None)
        request_class = str(raw_class).upper() if raw_class else plan.request_id.split("_", 1)[0].upper()
        if request_class not in {"S", "M", "L"}:
            request_class = "M"
        deadline_raw = getattr(value, "deadline_ms", None)
        deadline_ms = float(deadline_raw) if deadline_raw is not None else None
        priority = int(getattr(value, "priority", 0) or 0)
        arrival_ms = float(getattr(value, "arrival_ms", 0.0) or 0.0)
        t_single_raw = getattr(value, "profiled_optimal_latency_ms", None)
        t_single_ms = float(t_single_raw) if t_single_raw is not None else self._estimate_plan_t_single_ms(plan)
        return _RequestMeta(
            deadline_ms=deadline_ms,
            priority=priority,
            arrival_ms=arrival_ms,
            order=order,
            request_class=request_class,
            t_single_ms=max(1.0, t_single_ms),
        )

    def _estimate_plan_t_single_ms(self, plan: RequestExecutionPlan) -> float:
        total = 0.0
        sp1_groups = [
            group for group in self.topology.groups
            if int(group.parallel_spec.tp) == 1
            and int(group.parallel_spec.sp) == 1
            and int(group.parallel_spec.cfg) == 1
        ]
        if not sp1_groups:
            sp1_groups = [
                self._ensure_group_for_ranks(
                    ranks=(self._worker_ranks()[0],),
                    supported_task_kinds=self._STANDARD_TASK_KINDS,
                )
            ]
        group_id = sp1_groups[0].group_id
        for task in self._linear_task_sequence(plan):
            try:
                total += self._estimate_task_ms(task, group_id)
            except Exception:
                pass
        return max(1.0, total)

    @staticmethod
    def _linear_task_sequence(plan: RequestExecutionPlan) -> tuple[InferenceTask, ...]:
        tasks = dict(plan.tasks)
        remaining = set(tasks)
        out: list[InferenceTask] = []
        satisfied: set[str] = set()
        while remaining:
            progressed = False
            for task_id in sorted(remaining):
                task = tasks[task_id]
                if all(dependency in satisfied for dependency in task.dependencies):
                    out.append(task)
                    satisfied.add(task_id)
                    remaining.remove(task_id)
                    progressed = True
                    break
            if not progressed:
                raise ValueError(f"request {plan.request_id} task graph is not a DAG")
        return tuple(out)

    # ------------------------------------------------------------------
    # Resource helpers
    # ------------------------------------------------------------------

    def _safe_to_hold(self) -> bool:
        next_release = self._next_predicted_release_ms()
        return next_release is not None and next_release > self.now_ms + 1e-6

    def _next_predicted_release_ms(self) -> float | None:
        future = [
            finish_ms
            for finish_ms, _ranks, _group_id, _request_id in self.predicted_running.values()
            if finish_ms > self.now_ms + 1e-6
        ]
        return min(future) if future else None

    def _can_dispatch_ranks(self, ranks: tuple[int, ...]) -> bool:
        for rank in ranks:
            if self.outstanding_per_rank.get(rank, 0) > 0:
                return False
            if self.reshard_holds_by_rank.get(rank, 0) > 0:
                return False
        return True

    @staticmethod
    def _rank_mask(ranks: tuple[int, ...]) -> int:
        mask = 0
        for rank in ranks:
            mask |= 1 << int(rank)
        return mask

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
        group_id = f"pd_sp{sp_size}_r{'_'.join(str(rank) for rank in ranks)}"
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

    def _normalize_allowed_sp_sizes(self, allowed_sp_sizes: Iterable[int] | None) -> tuple[int, ...]:
        max_size = max(1, len(self.topology.workers))
        if allowed_sp_sizes is None:
            sizes = {
                int(group.parallel_spec.sp)
                for group in self.topology.groups
                if int(group.parallel_spec.tp) == 1 and int(group.parallel_spec.cfg) == 1
            }
            sizes.add(1)
            sizes.add(max_size)
        else:
            sizes = {int(size) for size in allowed_sp_sizes}
            sizes.add(1)
        valid = tuple(sorted((size for size in sizes if 1 <= size <= max_size), reverse=True))
        if not valid:
            raise ValueError("PrimalDualRollingPlannerPolicy needs at least one valid SP size")
        return valid

    def _reference_t_single_ms(self) -> float:
        values = sorted(meta.t_single_ms for meta in self.request_meta.values() if meta.t_single_ms > 0)
        if not values:
            return 1.0
        return max(1.0, values[len(values) // 2])

    @staticmethod
    def _t_single(meta: _RequestMeta) -> float:
        return max(1.0, float(meta.t_single_ms))

    # ------------------------------------------------------------------
    # Debug
    # ------------------------------------------------------------------

    def _debug_plan(
        self,
        plan: _ShadowState,
        *,
        committed: tuple[_PlanAction, ...],
        fallback: bool = False,
    ) -> None:
        if not self.debug:
            return
        chosen = [
            f"{action.request_id}:{action.task_kind.value}@{action.group_id}"
            for action in committed
        ]
        hold = ""
        if plan.hold is not None:
            hold = f" hold_until={plan.hold.until_ms:.3f} hold_reason={plan.hold.reason}"
        print(
            "[pd-roll] plan "
            f"t={self.now_ms:.3f} active={len(self.request_meta)} "
            f"running={len(self.predicted_running)} "
            f"lambda={{{', '.join(f'{k}:{v:.2f}' for k, v in sorted(self.lambda_by_class.items()))}}} "
            f"committed={chosen} fallback={int(fallback)}{hold}",
            flush=True,
        )


def _parse_class_float_map(raw: str, *, default: dict[str, float]) -> dict[str, float]:
    out = dict(default)
    raw = raw.strip()
    if not raw:
        return out
    for item in raw.split(","):
        if not item.strip():
            continue
        key, value = item.split(":", 1)
        out[key.strip().upper()] = float(value)
    return out


def _parse_sp_sizes(raw: str) -> tuple[int, ...] | None:
    raw = raw.strip()
    if not raw:
        return None
    return tuple(int(part.strip()) for part in raw.split(",") if part.strip())


def make_primal_dual_rolling_planner(topology, task_runtime_estimator):
    return PrimalDualRollingPlannerPolicy(
        topology=topology,
        task_runtime_estimator=task_runtime_estimator,
        horizon_steps=int(os.getenv("PD_ROLL_HORIZON_STEPS", "10")),
        beam_size=int(os.getenv("PD_ROLL_BEAM_SIZE", "12")),
        top_m_requests=int(os.getenv("PD_ROLL_TOP_M_REQUESTS", "8")),
        top_g_groups=int(os.getenv("PD_ROLL_TOP_G_GROUPS", "8")),
        reshard_ms=float(os.getenv("PD_ROLL_RESHARD_MS", os.getenv("STEP_PREEMPT_RESHARD_MS", "3"))),
        dual_step_size=float(os.getenv("PD_ROLL_DUAL_STEP", "4.0")),
        ewma_alpha=float(os.getenv("PD_ROLL_EWMA_ALPHA", "0.20")),
        target_miss_rate=_parse_class_float_map(
            os.getenv("PD_ROLL_TARGET_MISS", ""),
            default={"S": 0.05, "M": 0.05, "L": 0.10},
        ),
        initial_lambda=_parse_class_float_map(
            os.getenv("PD_ROLL_LAMBDA_INIT", ""),
            default={"S": 0.0, "M": 0.0, "L": 0.0},
        ),
        lambda_cap=float(os.getenv("PD_ROLL_LAMBDA_CAP", "80.0")),
        latency_weight=_parse_class_float_map(
            os.getenv("PD_ROLL_LATENCY_WEIGHT", ""),
            default={"S": 1.0, "M": 1.0, "L": 1.0},
        ),
        hold_penalty_weight=float(os.getenv("PD_ROLL_HOLD_PENALTY", "0.02")),
        reshard_penalty_weight=float(os.getenv("PD_ROLL_RESHARD_PENALTY", "0.0")),
        gpu_time_penalty_weight=float(os.getenv("PD_ROLL_GPU_TIME_PENALTY", "0.2")),
        pressure_threshold=float(os.getenv("PD_ROLL_PRESSURE_THRESHOLD", "1.0")),
        predictive_dual=str(os.getenv("PD_ROLL_PREDICTIVE_DUAL", "1")).strip().lower() in {"1", "true", "yes", "on"},
        predictive_dual_throttle_ms=float(os.getenv("PD_ROLL_PREDICTIVE_THROTTLE_MS", "500")),
        allowed_sp_sizes=_parse_sp_sizes(os.getenv("PD_ROLL_SP_SIZES", "")),
        final_edf_tail_states=int(os.getenv("PD_ROLL_FINAL_EDF_TAIL_STATES", "0")),
        batch_states_per_mask=int(os.getenv("PD_ROLL_BATCH_STATES_PER_MASK", "1")),
        batch_final_tail_states=int(os.getenv("PD_ROLL_BATCH_FINAL_TAIL_STATES", "0")),
        planner_mode=os.getenv("PD_ROLL_MODE", "batch").strip().lower() or "batch",
        debug=str(os.getenv("PD_ROLL_DEBUG", "")).strip().lower() in {"1", "true", "yes", "on"},
    )


def make_pd_rolling(topology, task_runtime_estimator):
    return make_primal_dual_rolling_planner(topology, task_runtime_estimator)
