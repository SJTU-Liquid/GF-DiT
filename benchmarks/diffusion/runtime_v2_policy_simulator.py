#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Offline policy simulator for runtime_v2 diffusion scheduling.

The simulator is intentionally CPU-only. It builds runtime_v2 task plans,
drives a SchedulerPolicy with virtual worker events, estimates task duration
from fitted cost-model JSON files, and writes a Chrome trace timeline.
"""

from __future__ import annotations

import argparse
import heapq
import importlib
import json
import math
import os
import random
import statistics
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from vllm_omni.diffusion.runtime_v2.adapters.common import (
    STANDARD_TASK_KINDS,
    build_chunked_dit_plan,
)
from vllm_omni.diffusion.runtime_v2.cost_model import (
    CostModelKey,
    CostModelStore,
    StageCostModel,
)
from vllm_omni.diffusion.runtime_v2.protocol import (
    ExecutionGroupSpec,
    InferenceTask,
    ParallelSpec,
    RequestExecutionPlan,
    RESHARD_PAYLOAD_DST_GROUP_ID,
    RESHARD_PAYLOAD_SRC_GROUP_ID,
    TaskKind,
    TaskStatus,
    WorkerEvent,
    WorkerEventKind,
)
from vllm_omni.diffusion.runtime_v2.scheduler import (
    FCFSSchedulerPolicy,
    SRTFSchedulerPolicy,
    TaskRuntimeEstimator,
)
from vllm_omni.diffusion.runtime_v2.topology import RuntimeTopology, WorkerSpec


@dataclass(frozen=True)
class SimRequest:
    request_id: str
    arrival_ms: float
    height: int
    width: int
    num_frames: int
    num_steps: int
    denoise_chunk_size: int
    text_seq_len: int
    priority: int = 0
    group_id: str | None = None
    # Optional metadata used by mixed-priority experiments. Step-preempt
    # policies read these via getattr so leaving them at None is safe for
    # existing single-class workloads.
    deadline_ms: float | None = None
    profiled_optimal_latency_ms: float | None = None
    request_class: str | None = None
    preemptible: bool | None = None
    do_true_cfg: bool | None = None
    latent_seq_len_override: int | None = None
    voxels_override: int | None = None

    @property
    def latent_seq_len(self) -> int:
        if self.latent_seq_len_override is not None:
            return max(1, int(self.latent_seq_len_override))
        spatial_tokens = max(1, (self.height // 16) * (self.width // 16))
        temporal_tokens = max(1, (self.num_frames + 3) // 4)
        return max(1, spatial_tokens * temporal_tokens)

    @property
    def voxels(self) -> int:
        if self.voxels_override is not None:
            return max(1, int(self.voxels_override))
        return max(1, self.height * self.width * self.num_frames)


@dataclass(order=True)
class RunningTask:
    end_ms: float
    sequence: int
    task_id: str
    group_id: str
    ranks: tuple[int, ...] = field(compare=False)
    launch_begin_ms: float = field(compare=False, default=0.0)
    launch_end_ms: float = field(compare=False, default=0.0)
    exec_begin_ms: float = field(compare=False, default=0.0)


SIM_TASK_LAUNCH_KIND = "task_launch"


@dataclass
class QueuedTask:
    enqueue_ms: float
    sequence: int
    task_id: str


@dataclass(order=True)
class SimEvent:
    time_ms: float
    sequence: int
    kind: str | WorkerEventKind = field(compare=False)
    request: SimRequest | None = field(default=None, compare=False)
    worker_event: WorkerEvent | None = field(default=None, compare=False)
    running_task: RunningTask | None = field(default=None, compare=False)


@dataclass
class TimelineEvent:
    request_id: str
    task_id: str
    kind: str
    group_id: str
    ranks: tuple[int, ...]
    start_ms: float
    end_ms: float
    args: dict[str, Any] = field(default_factory=dict)


@dataclass
class SimulationResult:
    timeline: list[TimelineEvent]
    request_arrival_ms: dict[str, float]
    request_finish_ms: dict[str, float]
    trace_events: list[dict[str, Any]]
    global_idle_ms: float
    global_idle_frac: float


class RuntimeV2PolicySimulator:
    def __init__(
        self,
        *,
        topology: RuntimeTopology,
        policy_factory: Callable[[RuntimeTopology, TaskRuntimeEstimator], Any],
        cost_models: CostModelStore,
        reshard_ms: float = 0.0,
        reshard_samples: list[float] | None = None,
        launch_end_before_exec_end_ms: float = 0.05,
        launch_overhead_ms: float | None = None,
        exec_start_delay_ms: float = 0.005,
        round_ms: float = 0.0,
        trace_color_by: str = "kind",
        debug_scheduler: bool = False,
    ) -> None:
        self.topology = topology
        self.cost_models = cost_models
        self.reshard_ms = float(reshard_ms)
        # Optional empirical reshard-duration distribution: when given, each
        # RESHARD task samples its duration (seeded) from these real per-rank
        # migration occupancies instead of the flat reshard_ms — lets us replay
        # the measured tail (p99) and ablate it.
        self._reshard_samples = list(reshard_samples) if reshard_samples else None
        self._reshard_model_enabled = self.reshard_ms > 0.0 or self._reshard_samples is not None
        self._reshard_rng = random.Random(int(os.environ.get("VLLM_RESHARD_SEED", "777")))
        if launch_overhead_ms is not None:
            launch_end_before_exec_end_ms = launch_overhead_ms
        self.launch_end_before_exec_end_ms = max(0.0, float(launch_end_before_exec_end_ms))
        self.exec_start_delay_ms = max(0.0, float(exec_start_delay_ms))
        if trace_color_by not in {"kind", "request"}:
            raise ValueError(f"trace_color_by must be 'kind' or 'request', got {trace_color_by!r}")
        self.trace_color_by = trace_color_by
        self.debug_scheduler = bool(debug_scheduler)
        # Optional multiplicative execution-time jitter to probe how sensitive a
        # policy's per-tick re-planning is to completion-time variance (the real
        # runtime has GPU/GIL/migration jitter the deterministic cost model lacks).
        # Each task duration is scaled by 1 + U(-frac, +frac). Env-gated so it
        # stays off for normal alignment runs.
        self.exec_jitter_frac = max(0.0, float(os.environ.get("VLLM_SIM_JITTER", "0") or 0.0))
        self._jitter_rng = random.Random(int(os.environ.get("VLLM_SIM_JITTER_SEED", "12345")))
        self.policy = policy_factory(topology, self._policy_estimate_ms)

        # Fixed-interval round scheduling (faithful TetriServe / Route B). When
        # round_ms > 0 and the policy exposes on_round_tick, the simulator wakes
        # the policy every round_ms so it can re-plan only at round boundaries.
        self.round_ms = max(0.0, float(round_ms))
        policy_wants_rounds = hasattr(self.policy, "on_round_tick")
        if policy_wants_rounds and self.round_ms <= 0.0:
            # A round policy with no round length never opens its dispatch gate,
            # so it would silently complete zero requests. Fail fast instead.
            raise ValueError(
                f"policy {type(self.policy).__name__} requires a positive round "
                "length: pass --round-ms > 0 (round_ms=) for round-based policies"
            )
        self._round_mode = self.round_ms > 0.0 and policy_wants_rounds
        if self._round_mode:
            # Let the round policy know the boundary spacing (for within-round
            # continuation decisions).
            try:
                self.policy.round_ms = self.round_ms
            except AttributeError:
                pass
        self._total_requests = 0
        self._round_ticks_emitted = 0

        self.plans: dict[str, RequestExecutionPlan] = {}
        self.task_index: dict[str, InferenceTask] = {}
        self.pending_dependencies: dict[str, int] = {}
        self.dependents: dict[str, list[str]] = {}
        self.event_heap: list[SimEvent] = []
        self.running: dict[str, RunningTask] = {}
        self.worker_queues: dict[str, deque[QueuedTask]] = defaultdict(deque)
        self.rank_busy: dict[int, str | None] = {worker.worker_rank: None for worker in topology.workers}
        self.rank_exec_free_ms: dict[int, float] = {worker.worker_rank: 0.0 for worker in topology.workers}
        self.timeline: list[TimelineEvent] = []
        self.request_arrival_ms: dict[str, float] = {}
        self.request_finish_ms: dict[str, float] = {}
        self.task_completed_group: dict[str, str] = {}
        self._launch_finished_task_ids: set[str] = set()
        self._exec_finished_task_ids: set[str] = set()
        self._request_finish_enqueued: set[str] = set()
        self._lazy_dispatch_warnings: set[tuple[float, tuple[str, ...]]] = set()
        self._sequence = 0
        # When a DiT chunk switches SP groups, we inject a RESHARD task that
        # occupies src∪dst ranks for reshard_ms (gated on the barrier) and hold
        # the DiT until it completes. Mirrors the real runtime's migration,
        # which the old per-step penalty under-modeled (it only delayed the one
        # migrating request and never occupied the src ranks).
        self._reshard_pending: dict[str, InferenceTask] = {}
        self._reshard_seq = 0
        # task_ids of RESHARDs the SIMULATOR injected (vs explicit RESHARDs the
        # policy dispatches from the plan). Only synthetic ones are hidden from
        # the policy — explicit reshards are real policy-tracked tasks whose
        # TASK_EXEC_END the policy needs to release its reshard holds.
        self._synthetic_reshard_ids: set[str] = set()

    def run(self, requests: Iterable[SimRequest]) -> SimulationResult:
        arrivals = sorted(requests, key=lambda req: (req.arrival_ms, req.request_id))
        now_ms = 0.0
        for request in arrivals:
            self._push_event(time_ms=request.arrival_ms, kind="arrival", request=request)
        self._total_requests = len(arrivals)
        if self._round_mode and arrivals:
            # First round boundary at the first arrival; subsequent ticks are
            # rescheduled as each one fires (while work remains).
            self._push_event(time_ms=arrivals[0].arrival_ms, kind="round_tick")

        while self.event_heap or self.running or self._has_queued_tasks():
            if self.event_heap:
                next_event_time = self.event_heap[0].time_ms
                if (
                    self._has_queued_tasks()
                    and self._all_ranks_idle()
                    and next_event_time > now_ms + 1e-9
                ):
                    started = self._poll_workers(now_ms)
                    if not started:
                        self._assert_no_idle_if_work_exists(now_ms)
                        self._debug_idle_reason(now_ms)
                        break
                    continue
                now_ms = next_event_time
            else:
                started = self._poll_workers(now_ms)
                if not started:
                    self._assert_no_idle_if_work_exists(now_ms)
                    self._debug_idle_reason(now_ms)
                    break
                continue

            while True:
                processed = self._process_events_at(now_ms)
                started = self._poll_workers(now_ms)
                if not processed and not started:
                    self._assert_no_idle_if_work_exists(now_ms)
                    self._debug_idle_reason(now_ms)
                    break
                if not self._has_event_at(now_ms) and not started:
                    self._debug_idle_reason(now_ms)
                    break

        global_idle_ms, global_idle_frac = compute_global_idle(
            self.timeline,
            start_ms=min(self.request_arrival_ms.values(), default=0.0),
            end_ms=max(self.request_finish_ms.values(), default=0.0),
        )
        trace_events = build_chrome_trace_events(
            self.topology,
            self.timeline,
            color_by=self.trace_color_by,
        )
        return SimulationResult(
            timeline=self.timeline,
            request_arrival_ms=dict(self.request_arrival_ms),
            request_finish_ms=dict(self.request_finish_ms),
            trace_events=trace_events,
            global_idle_ms=global_idle_ms,
            global_idle_frac=global_idle_frac,
        )

    def _process_events_at(self, now_ms: float) -> bool:
        processed = False
        to_dispatch: list[InferenceTask] = []
        while self._has_event_at(now_ms):
            events: list[SimEvent] = []
            while self._has_event_at(now_ms):
                events.append(heapq.heappop(self.event_heap))
            processed = True
            # Process round_tick LAST among same-timestamp events. Otherwise a
            # tick scheduled earlier (smaller sequence) can fire before a
            # same-time TASK_EXEC_END, advancing the round window before the
            # completing request frees its ranks -- letting it cross the boundary
            # via within-round continuation instead of being re-admitted by the
            # boundary planner over the freed pool.
            events.sort(key=lambda e: e.kind == "round_tick")
            for event in events:
                if event.kind == "arrival":
                    if event.request is None:
                        raise ValueError("arrival event missing request")
                    to_dispatch.extend(self._submit_request(event.request, now_ms))
                    continue
                if event.kind == "round_tick":
                    to_dispatch.extend(self._handle_round_tick(now_ms))
                    continue
                if event.worker_event is None:
                    raise ValueError(f"worker event {event.kind!r} missing payload")
                to_dispatch.extend(self._handle_worker_event(event, now_ms))

        if not processed:
            return False
        self._dispatch(to_dispatch, now_ms)
        return True

    def _submit_request(self, request: SimRequest, now_ms: float) -> list[InferenceTask]:
        self._debug(
            now_ms,
            "arrival",
            request_id=request.request_id,
            shape=f"{request.height}x{request.width}x{request.num_frames}",
            steps=request.num_steps,
        )
        plan = self._build_request_plan(request)
        self.request_arrival_ms[request.request_id] = request.arrival_ms
        self.plans[plan.request_id] = plan
        for task in plan.tasks.values():
            self.task_index[task.task_id] = task
            self.pending_dependencies[task.task_id] = len(task.dependencies)
            for dependency in task.dependencies:
                self.dependents.setdefault(dependency, []).append(task.task_id)
        return self._policy_returned(
            now_ms,
            "on_request_submitted",
            self.policy.on_request_submitted(plan),
        )

    def _build_request_plan(self, request: SimRequest) -> RequestExecutionPlan:
        build_hook = getattr(self.policy, "build_sim_plan", None)
        if callable(build_hook):
            return build_hook(request=request, default_builder=build_sim_plan)
        return build_sim_plan(request)

    def _handle_round_tick(self, now_ms: float) -> list[InferenceTask]:
        self._round_ticks_emitted += 1
        to_dispatch = self._policy_returned(
            now_ms, "on_round_tick", self.policy.on_round_tick(now_ms)
        )
        # Reschedule the next boundary while requests remain unfinished. The
        # tick cap is a safety net against a misbehaving policy that never
        # makes progress (the round window can be far shorter than the workload).
        if (
            len(self.request_finish_ms) < self._total_requests
            and self._round_ticks_emitted < 5_000_000
        ):
            self._push_event(time_ms=now_ms + self.round_ms, kind="round_tick")
        return to_dispatch

    def _handle_worker_event(self, event: SimEvent, now_ms: float) -> list[InferenceTask]:
        worker_event = event.worker_event
        if worker_event is None:
            raise ValueError("worker event missing payload")
        if worker_event.kind == WorkerEventKind.REQUEST_FINISHED:
            self._handle_request_finished(worker_event, now_ms)
            return self._policy_returned(
                now_ms,
                f"on_worker_event:{worker_event.kind.value}",
                self.policy.on_worker_event(worker_event),
            )

        if worker_event.kind == WorkerEventKind.TASK_LAUNCH_END:
            if event.running_task is None:
                raise ValueError(f"TASK_LAUNCH_END event missing running task: {worker_event.task_id}")
            to_dispatch = self._handle_task_launch_end(event.running_task, now_ms)
            if not self._is_synthetic_reshard(event.running_task.task_id):
                to_dispatch.extend(
                    self._policy_returned(
                        now_ms,
                        f"on_worker_event:{worker_event.kind.value}",
                        self.policy.on_worker_event(worker_event),
                    )
                )
            to_dispatch.extend(self._drain_policy_after_worker_event(now_ms, worker_event.kind))
            return to_dispatch

        if worker_event.kind == WorkerEventKind.TASK_EXEC_END:
            if event.running_task is None:
                raise ValueError(f"TASK_EXEC_END event missing running task: {worker_event.task_id}")
            to_dispatch = self._handle_task_exec_end(event.running_task, now_ms)
            # Synthetic RESHARD tasks are injected by the simulator, not the
            # policy; forwarding their completion would make dynamic policies
            # (e.g. edf_greedy/pd_rolling) decrement request_inflight and update
            # request_last_group for a task they never dispatched.
            if not self._is_synthetic_reshard(event.running_task.task_id):
                to_dispatch.extend(
                    self._policy_returned(
                        now_ms,
                        f"on_worker_event:{worker_event.kind.value}",
                        self.policy.on_worker_event(worker_event),
                    )
                )
            return to_dispatch

        to_dispatch = self._policy_returned(
            now_ms,
            f"on_worker_event:{worker_event.kind.value}",
            self.policy.on_worker_event(worker_event),
        )
        return to_dispatch

    def _is_synthetic_reshard(self, task_id: str) -> bool:
        return task_id in self._synthetic_reshard_ids

    def _dispatch(self, tasks: Iterable[InferenceTask], now_ms: float) -> None:
        for task in tasks:
            src_group_id = self._switch_src_group(task) if self._reshard_model_enabled else None
            if src_group_id is not None:
                self._emit_reshard(task, src_group_id, now_ms)
            else:
                self._enqueue_task(task, now_ms)

    def _enqueue_task(self, task: InferenceTask, now_ms: float) -> None:
        task.status = TaskStatus.DISPATCHED
        group_id, _ = self._task_resource(task)
        self._sequence += 1
        self.worker_queues[group_id].append(
            QueuedTask(enqueue_ms=now_ms, sequence=self._sequence, task_id=task.task_id)
        )
        self._debug(
            now_ms,
            "dispatch_enqueue",
            task_id=task.task_id,
            group_id=group_id,
            queue_depth=len(self.worker_queues[group_id]),
        )

    def _switch_src_group(self, task: InferenceTask) -> str | None:
        """If this task runs on a different group than the one its input
        artifact was produced on, return that producing (src) group id — i.e. a
        cross-group state migration is required before this task can run. Covers
        DiT-to-DiT SP switches AND aux<->DiT stage-boundary moves (text->dit
        migrates state_text, dit->vae migrates state_denoised), mirroring the
        real runtime's implicit migrations at every cross-group dependency.
        RESHARD tasks themselves never migrate."""
        if task.kind == TaskKind.RESHARD:
            return None
        try:
            dst_group_id, _ = self._task_resource(task)
        except ValueError:
            return None
        for dependency in task.dependencies:
            src = self.task_completed_group.get(dependency)
            if src is not None and src != dst_group_id:
                return src
        return None

    def _emit_reshard(self, dit_task: InferenceTask, src_group_id: str, now_ms: float) -> None:
        """Inject a RESHARD task (occupies src∪dst ranks for reshard_ms, barrier-
        gated) and hold the DiT chunk until the reshard completes."""
        dst_group_id, _ = self._task_resource(dit_task)
        self._reshard_seq += 1
        reshard_id = f"{dit_task.task_id}::reshard{self._reshard_seq}"
        reshard = InferenceTask(
            task_id=reshard_id,
            request_id=dit_task.request_id,
            kind=TaskKind.RESHARD,
            group_id=dst_group_id,
            parallel_spec=dit_task.parallel_spec,
            payload={
                RESHARD_PAYLOAD_SRC_GROUP_ID: src_group_id,
                RESHARD_PAYLOAD_DST_GROUP_ID: dst_group_id,
            },
        )
        self.task_index[reshard_id] = reshard
        self._reshard_pending[reshard_id] = dit_task
        self._synthetic_reshard_ids.add(reshard_id)
        self._enqueue_task(reshard, now_ms)

    def _poll_workers(self, now_ms: float) -> bool:
        started_any = False
        while True:
            candidate = self._next_startable_task()
            if candidate is None:
                return started_any
            group_id, queued_task, ranks = candidate
            self.worker_queues[group_id].popleft()
            task = self.task_index[queued_task.task_id]
            task.status = TaskStatus.RUNNING
            duration_ms = self._task_duration_ms(task, group_id)
            exec_start_delay_ms = self._task_exec_start_delay_ms(duration_ms)
            stream_ready_ms = max((self.rank_exec_free_ms[rank] for rank in ranks), default=now_ms)
            exec_begin_ms = max(now_ms + exec_start_delay_ms, stream_ready_ms)
            exec_end_ms = exec_begin_ms + duration_ms
            launch_end_tail_ms = self._task_launch_end_tail_ms(duration_ms)
            launch_end_ms = max(now_ms, exec_end_ms - launch_end_tail_ms)
            for rank in ranks:
                self.rank_exec_free_ms[rank] = exec_end_ms
            for rank in ranks:
                self.rank_busy[rank] = task.task_id
            self._debug(
                now_ms,
                "worker_start",
                task_id=task.task_id,
                group_id=group_id,
                ranks=list(ranks),
                duration_ms=f"{duration_ms:.3f}",
                launch_end_ms=f"{launch_end_ms:.3f}",
                exec_begin_ms=f"{exec_begin_ms:.3f}",
                exec_end_ms=f"{exec_end_ms:.3f}",
                queue_wait_ms=f"{now_ms - queued_task.enqueue_ms:.3f}",
            )
            started_any = True
            self.timeline.append(
                TimelineEvent(
                    request_id=task.request_id,
                    task_id=task.task_id,
                    kind=SIM_TASK_LAUNCH_KIND,
                    group_id=group_id,
                    ranks=ranks,
                    start_ms=now_ms,
                    end_ms=launch_end_ms,
                    args={
                        **self._trace_args(task=task, group_id=group_id, duration_ms=duration_ms),
                        "phase": "launch",
                        "launch_duration_ms": launch_end_ms - now_ms,
                        "launch_end_before_exec_end_ms": exec_end_ms - launch_end_ms,
                        "exec_begin_ms": exec_begin_ms,
                        "exec_end_ms": exec_end_ms,
                        "enqueue_ms": queued_task.enqueue_ms,
                        "queue_wait_ms": now_ms - queued_task.enqueue_ms,
                        "required_ranks": list(ranks),
                    },
                )
            )
            self.timeline.append(
                TimelineEvent(
                    request_id=task.request_id,
                    task_id=task.task_id,
                    kind=task.kind.value,
                    group_id=group_id,
                    ranks=ranks,
                    start_ms=exec_begin_ms,
                    end_ms=exec_end_ms,
                    args={
                        **self._trace_args(task=task, group_id=group_id, duration_ms=duration_ms),
                        "phase": "exec",
                        "launch_begin_ms": now_ms,
                        "launch_end_ms": launch_end_ms,
                        "enqueue_ms": queued_task.enqueue_ms,
                        "queue_wait_ms": now_ms - queued_task.enqueue_ms,
                        "required_ranks": list(ranks),
                    },
                )
            )
            running = RunningTask(
                end_ms=exec_end_ms,
                sequence=self._next_sequence(),
                task_id=task.task_id,
                group_id=group_id,
                ranks=ranks,
                launch_begin_ms=now_ms,
                launch_end_ms=launch_end_ms,
                exec_begin_ms=exec_begin_ms,
            )
            self.running[task.task_id] = running
            self._push_worker_event(
                time_ms=now_ms,
                task=task,
                kind=WorkerEventKind.TASK_LAUNCH_BEGIN,
                group_id=group_id,
            )
            self._push_worker_event(
                time_ms=launch_end_ms,
                task=task,
                kind=WorkerEventKind.TASK_LAUNCH_END,
                group_id=group_id,
                running_task=running,
            )
            self._push_worker_event(
                time_ms=exec_begin_ms,
                task=task,
                kind=WorkerEventKind.TASK_EXEC_BEGIN,
                group_id=group_id,
            )
            self._push_worker_event(
                time_ms=exec_end_ms,
                task=task,
                kind=WorkerEventKind.TASK_EXEC_END,
                group_id=group_id,
                running_task=running,
            )

    def _next_startable_task(self) -> tuple[str, QueuedTask, tuple[int, ...]] | None:
        candidates: list[tuple[float, int, str, QueuedTask, tuple[int, ...]]] = []
        for group_id, queue in self.worker_queues.items():
            if not queue:
                continue
            queued_task = queue[0]
            task = self.task_index[queued_task.task_id]
            _, ranks = self._task_resource(task)
            if all(self.rank_busy[rank] is None for rank in ranks):
                candidates.append((queued_task.enqueue_ms, queued_task.sequence, group_id, queued_task, ranks))
        if not candidates:
            return None
        _, _, group_id, queued_task, ranks = min(candidates)
        return group_id, queued_task, ranks

    def _has_queued_tasks(self) -> bool:
        return any(queue for queue in self.worker_queues.values())

    def _assert_no_idle_if_work_exists(self, now_ms: float) -> None:
        if not self._has_queued_tasks() or not self._all_ranks_idle():
            return
        queue_depths = {group_id: len(queue) for group_id, queue in self.worker_queues.items() if queue}
        raise AssertionError(
            f"worker idle while work is buffered at {now_ms:.6f} ms: "
            f"worker_queues={queue_depths}"
        )

    def _handle_task_launch_end(self, running: RunningTask, now_ms: float) -> list[InferenceTask]:
        task = self.task_index[running.task_id]
        if task.task_id in self._launch_finished_task_ids:
            return []
        self._launch_finished_task_ids.add(task.task_id)
        if task.kind == TaskKind.RESHARD:
            # A RESHARD occupies the synthetic "src->dst" resource, but its
            # OUTPUT lands on the dst group. Record dst so a task depending on
            # this reshard (explicit plan reshards) doesn't see a phantom
            # "src->dst" producer group and spuriously re-migrate.
            self.task_completed_group[task.task_id] = str(task.payload[RESHARD_PAYLOAD_DST_GROUP_ID])
        else:
            self.task_completed_group[task.task_id] = running.group_id
        for rank in running.ranks:
            if self.rank_busy[rank] == task.task_id:
                self.rank_busy[rank] = None
        self._debug(
            now_ms,
            "worker_launch_end",
            task_id=task.task_id,
            group_id=running.group_id,
            ranks=list(running.ranks),
        )
        to_dispatch: list[InferenceTask] = []
        to_dispatch.extend(self._mark_dependencies_runnable(task, now_ms))
        return to_dispatch

    def _handle_task_exec_end(self, running: RunningTask, now_ms: float) -> list[InferenceTask]:
        task = self.task_index[running.task_id]
        self.running.pop(task.task_id, None)
        self._exec_finished_task_ids.add(task.task_id)
        task.status = TaskStatus.FINISHED
        for rank in running.ranks:
            if self.rank_busy[rank] == task.task_id:
                self.rank_busy[rank] = None
        # A finished RESHARD releases the DiT chunk it was gating; enqueue it now
        # that src∪dst ranks are free again.
        held_dit = self._reshard_pending.pop(task.task_id, None)
        if held_dit is not None:
            self._enqueue_task(held_dit, now_ms)
        group_id = running.group_id
        self._debug(
            now_ms,
            "worker_finish",
            task_id=task.task_id,
            group_id=group_id,
            ranks=list(running.ranks),
            phase="exec_end",
        )
        to_dispatch: list[InferenceTask] = []
        if self._request_is_finished(task.request_id):
            self._push_request_finished_event(request_id=task.request_id, group_id=group_id, now_ms=now_ms)
        return to_dispatch

    def _mark_dependencies_runnable(self, task: InferenceTask, now_ms: float) -> list[InferenceTask]:
        newly_runnable: list[InferenceTask] = []
        for dependent_id in self.dependents.get(task.task_id, []):
            self.pending_dependencies[dependent_id] -= 1
            if self.pending_dependencies[dependent_id] == 0:
                dependent = self.task_index[dependent_id]
                newly_runnable.append(dependent)
                self._debug(
                    now_ms,
                    "dependency_unlock",
                    task_id=dependent.task_id,
                    dependency=task.task_id,
                )
        if not newly_runnable:
            return []
        return self._policy_returned(
            now_ms,
            "on_tasks_runnable",
            self.policy.on_tasks_runnable(newly_runnable),
        )

    def _debug(self, now_ms: float, action: str, **fields: Any) -> None:
        if not self.debug_scheduler:
            return
        suffix = " ".join(f"{key}={value}" for key, value in fields.items())
        print(f"[sim {now_ms:.3f} ms] {action} {suffix}".rstrip())

    def _next_sequence(self) -> int:
        self._sequence += 1
        return self._sequence

    def _push_event(
        self,
        *,
        time_ms: float,
        kind: str | WorkerEventKind,
        request: SimRequest | None = None,
        worker_event: WorkerEvent | None = None,
        running_task: RunningTask | None = None,
    ) -> None:
        heapq.heappush(
            self.event_heap,
            SimEvent(
                time_ms=time_ms,
                sequence=self._next_sequence(),
                kind=kind,
                request=request,
                worker_event=worker_event,
                running_task=running_task,
            ),
        )

    def _push_worker_event(
        self,
        *,
        time_ms: float,
        task: InferenceTask,
        kind: WorkerEventKind,
        group_id: str,
        running_task: RunningTask | None = None,
    ) -> None:
        self._push_event(
            time_ms=time_ms,
            kind=kind,
            worker_event=self._worker_event(
                task=task,
                kind=kind,
                group_id=group_id,
                now_ms=time_ms,
            ),
            running_task=running_task,
        )

    def _push_request_finished_event(self, *, request_id: str, group_id: str, now_ms: float) -> None:
        if request_id in self._request_finish_enqueued or request_id in self.request_finish_ms:
            return
        self._request_finish_enqueued.add(request_id)
        self._push_event(
            time_ms=now_ms,
            kind=WorkerEventKind.REQUEST_FINISHED,
            worker_event=WorkerEvent(
                event_id=f"{request_id}:request_finished:{now_ms:.6f}",
                task_id="",
                request_id=request_id,
                group_id=group_id,
                worker_rank=-1,
                kind=WorkerEventKind.REQUEST_FINISHED,
                timestamp_ns=int(now_ms * 1_000_000),
            ),
        )

    def _has_event_at(self, now_ms: float) -> bool:
        return bool(self.event_heap) and self.event_heap[0].time_ms <= now_ms + 1e-9

    def _policy_returned(
        self,
        now_ms: float,
        source: str,
        tasks: Iterable[InferenceTask],
    ) -> list[InferenceTask]:
        returned = list(tasks)
        self._debug(
            now_ms,
            "policy_returned",
            source=source,
            task_ids=[task.task_id for task in returned],
        )
        return returned

    def _handle_request_finished(self, event: WorkerEvent, now_ms: float) -> None:
        self.request_finish_ms[event.request_id] = now_ms
        self._debug(
            now_ms,
            "request_finish",
            request_id=event.request_id,
            group_id=event.group_id,
        )

    def _request_is_finished(self, request_id: str) -> bool:
        plan = self.plans[request_id]
        if request_id in self._request_finish_enqueued or request_id in self.request_finish_ms:
            return False
        return all(task_id in self._exec_finished_task_ids for task_id in plan.terminal_task_ids)

    def _drain_policy_after_worker_event(
        self,
        now_ms: float,
        kind: WorkerEventKind,
    ) -> list[InferenceTask]:
        # Some policies use worker events to clear launch throttles and only
        # produce dispatchable work from on_tasks_runnable().  Give them one
        # empty runnable tick after the event so newly-runnable continuations
        # can be dispatched at the same simulator time.
        ready_task_ids = self._ready_task_ids()
        if not ready_task_ids:
            return []
        return self._policy_returned(
            now_ms,
            f"on_tasks_runnable:drain_after_{kind.value}",
            self.policy.on_tasks_runnable(()),
        )

    def _all_ranks_idle(self) -> bool:
        return all(task_id is None for task_id in self.rank_busy.values())

    def _ready_task_ids(self) -> list[str]:
        queued = {queued_task.task_id for queue in self.worker_queues.values() for queued_task in queue}
        return sorted(
            task_id
            for task_id, task in self.task_index.items()
            if task.status == TaskStatus.READY and task_id not in queued
        )

    def _blocked_task_ids(self) -> list[str]:
        return sorted(
            task_id
            for task_id, task in self.task_index.items()
            if task.status == TaskStatus.PENDING and self.pending_dependencies.get(task_id, 0) > 0
        )

    def _debug_idle_reason(self, now_ms: float) -> None:
        if not self.debug_scheduler or self.running or not self._all_ranks_idle():
            return
        if self._has_queued_tasks():
            queue_depths = {group_id: len(queue) for group_id, queue in self.worker_queues.items() if queue}
            self._debug(now_ms, "global_idle", reason="worker_queue_buffered", worker_queues=queue_depths)
            return
        ready_task_ids = self._ready_task_ids()
        if ready_task_ids:
            warning_key = (round(now_ms, 9), tuple(ready_task_ids))
            if warning_key not in self._lazy_dispatch_warnings:
                self._lazy_dispatch_warnings.add(warning_key)
                self._debug(
                    now_ms,
                    "debug_warning",
                    reason="policy_lazy_dispatch",
                    ready_task_ids=ready_task_ids,
                )
            return
        blocked_task_ids = self._blocked_task_ids()
        if blocked_task_ids:
            self._debug(
                now_ms,
                "global_idle",
                reason="dependencies_unmet",
                blocked_task_count=len(blocked_task_ids),
            )
            return
        if self.event_heap:
            self._debug(
                now_ms,
                "global_idle",
                reason="worker_queue_empty",
                next_event_ms=f"{self.event_heap[0].time_ms:.3f}",
            )

    def _task_resource(self, task: InferenceTask) -> tuple[str, tuple[int, ...]]:
        if task.kind == TaskKind.RESHARD:
            src = str(task.payload[RESHARD_PAYLOAD_SRC_GROUP_ID])
            dst = str(task.payload[RESHARD_PAYLOAD_DST_GROUP_ID])
            ranks = tuple(sorted(set(self.topology.get_group(src).ranks) | set(self.topology.get_group(dst).ranks)))
            return f"{src}->{dst}", ranks
        if task.group_id is None:
            raise ValueError(f"task {task.task_id} has no group_id after policy dispatch")
        group = self.topology.get_group(task.group_id)
        # Single-rank stages: text encoder, VAE decode and host-side finalize
        # do not benefit from sequence parallelism. Modeling them as occupying
        # the entire SP group would falsely block (sp-1) idle ranks during
        # those tens-of-ms / few-seconds tasks, hurting cross-request
        # concurrency in policies that bind a request to a single SP group.
        if task.kind in (TaskKind.TEXT_ENCODE, TaskKind.VAE_DECODE, TaskKind.FINALIZE):
            return group.group_id, (int(group.ranks[0]),)
        return group.group_id, tuple(group.ranks)

    def _task_duration_ms(self, task: InferenceTask, group_id: str) -> float:
        if task.kind == TaskKind.RESHARD:
            if self._reshard_samples:
                return max(0.0, self._reshard_rng.choice(self._reshard_samples))
            return max(0.0, self.reshard_ms)
        group = self.topology.get_group(group_id)
        model = self.cost_models.for_group(group)
        stage = str(task.payload.get("cost_model_stage") or task.kind.value)
        features = dict(task.payload.get("sim_features", {}))
        x_name = model.stage_x_name(stage)
        x = float(features[x_name])
        base = model.estimate_ms(stage, x)
        if task.kind == TaskKind.DIT_STEP_CHUNK:
            base *= float(features.get("chunk_steps", 1))
        if self.exec_jitter_frac > 0.0:
            base *= 1.0 + self._jitter_rng.uniform(-self.exec_jitter_frac, self.exec_jitter_frac)
            base = max(0.0, base)
        # Migration cost is now modeled explicitly as a RESHARD task that
        # occupies src∪dst ranks for reshard_ms (see _emit_reshard), so the DiT
        # chunk itself runs at its plain cost-model duration.
        return base

    def _task_launch_end_tail_ms(self, duration_ms: float) -> float:
        duration_ms = max(0.0, float(duration_ms))
        if duration_ms <= 0.0 or self.launch_end_before_exec_end_ms <= 0.0:
            return 0.0
        # CUDA eager launch quickly starts GPU work, then CPU enqueue can stay
        # busy until the bounded stream queue drains near exec completion.
        return min(self.launch_end_before_exec_end_ms, duration_ms)

    def _task_exec_start_delay_ms(self, duration_ms: float) -> float:
        duration_ms = max(0.0, float(duration_ms))
        if duration_ms <= 0.0 or self.exec_start_delay_ms <= 0.0:
            return 0.0
        return min(self.exec_start_delay_ms, duration_ms)

    def _policy_estimate_ms(self, parallel_spec: ParallelSpec, seq_len: int, request_stage: str) -> float:
        # Look up by parallel_spec directly instead of walking topology.groups.
        # With ``--topology-style minimal`` the SP2/SP4 execution groups don't
        # exist until edf_greedy materializes them on demand via
        # ``_ensure_group_for_ranks``; the cost-aware SP picker calls this
        # estimator BEFORE creating those groups, so the old topology-scan
        # raised KeyError on the very first dispatch. Mirror the production
        # estimator (`vllm_omni/.../runner.py:_make_cost_model_estimator`):
        # return 0.0 when the cost model has no entry for ``parallel_spec``,
        # so policies can degrade gracefully (best-fit treats 0 as "deadline
        # always met" → smallest sp; the edf_greedy cost picker treats 0 as
        # "no estimate" and sorts those SPs last via its own check).
        try:
            model = self.cost_models.for_parallel_spec(parallel_spec)
        except KeyError:
            return 0.0
        return model.estimate_ms(str(request_stage), float(seq_len))

    def _trace_args(self, *, task: InferenceTask, group_id: str, duration_ms: float) -> dict[str, Any]:
        args = {
            "request_id": task.request_id,
            "task_id": task.task_id,
            "kind": task.kind.value,
            "group_id": group_id,
            "duration_ms": duration_ms,
        }
        if task.step_range is not None:
            args["step_start"] = task.step_range.start
            args["step_end"] = task.step_range.end
        if task.kind != TaskKind.RESHARD and task.group_id is not None:
            model = self.cost_models.for_group(self.topology.get_group(task.group_id))
            stage = str(task.payload.get("cost_model_stage") or task.kind.value)
            args["cost_model"] = str(model.source_path)
            args["x_name"] = model.stage_x_name(stage)
            args["cost_model_stage"] = stage
        args.update(dict(task.payload.get("sim_features", {})))
        return args

    @staticmethod
    def _worker_event(
        *,
        task: InferenceTask,
        kind: WorkerEventKind,
        group_id: str,
        now_ms: float,
    ) -> WorkerEvent:
        return WorkerEvent(
            event_id=f"{task.task_id}:{kind.value}:{now_ms:.6f}",
            task_id=task.task_id,
            request_id=task.request_id,
            group_id=group_id,
            worker_rank=-1,
            kind=kind,
            timestamp_ns=int(now_ms * 1_000_000),
        )


def build_sim_plan(
    request: SimRequest,
    *,
    group_id: str | None = None,
    stage_group_ids: Mapping[str, str] | None = None,
    dit_step_schedule: tuple[Mapping[str, Any], ...] | None = None,
) -> RequestExecutionPlan:
    plan = build_chunked_dit_plan(
        request_id=request.request_id,
        request_value=request,
        request_type="runtime_v2_policy_sim",
        num_steps=request.num_steps,
        chunk_size=request.denoise_chunk_size,
        priority=request.priority,
        group_id=request.group_id if group_id is None else group_id,
        text_seq_len=request.text_seq_len,
        latent_seq_len=request.latent_seq_len,
        state_codec_id="sim_state",
        decoded_codec_id=None,
        stage_group_ids=stage_group_ids,
        dit_step_schedule=dit_step_schedule,
        dit_step_do_true_cfg=(
            (bool(request.do_true_cfg),) * request.num_steps
            if request.do_true_cfg is not None
            else None
        ),
        metadata={
            "height": request.height,
            "width": request.width,
            "num_frames": request.num_frames,
            "num_steps": request.num_steps,
            "denoise_chunk_size": request.denoise_chunk_size,
            "text_seq_len": request.text_seq_len,
            "latent_seq_len": request.latent_seq_len,
            "voxels": request.voxels,
        },
    )
    for task in plan.tasks.values():
        features = _task_features(request, task)
        payload = dict(task.payload)
        payload["sim_features"] = features
        payload["seq_len"] = int(features[_stage_default_x_name(task.kind)])
        task.payload = payload
    return plan


def _task_features(request: SimRequest, task: InferenceTask) -> dict[str, int]:
    chunk_steps = 1
    if task.step_range is not None:
        chunk_steps = task.step_range.end - task.step_range.start
    return {
        "text_seq_len": request.text_seq_len,
        "latent_seq_len": request.latent_seq_len,
        "num_steps": request.num_steps,
        "voxels": request.voxels,
        "ones": 1,
        "chunk_steps": max(1, chunk_steps),
        "height": request.height,
        "width": request.width,
        "num_frames": request.num_frames,
    }


def _stage_default_x_name(kind: TaskKind) -> str:
    if kind == TaskKind.TEXT_ENCODE:
        return "text_seq_len"
    if kind in (TaskKind.DIT_PREPARE, TaskKind.DIT_STEP_CHUNK):
        return "latent_seq_len"
    if kind == TaskKind.TIMESTEP_PREPARE:
        return "num_steps"
    if kind == TaskKind.VAE_DECODE:
        return "voxels"
    if kind == TaskKind.FINALIZE:
        return "ones"
    return "ones"


def build_topology(args: argparse.Namespace, cost_models: CostModelStore) -> RuntimeTopology:
    groups_json = None
    group_sizes = None
    if args.groups_json:
        groups_json = json.loads(args.groups_json)
    elif args.group_sizes:
        group_sizes = args.group_sizes
    else:
        groups_json = [_default_group_json(args)]

    worker_count = int(args.world_size or _worker_count(groups_json=groups_json, group_sizes=group_sizes))
    workers = tuple(WorkerSpec(worker_rank=i, device_id=i) for i in range(worker_count))
    if groups_json is not None:
        groups = _build_groups_from_groups_json(groups_json)
    elif group_sizes is not None:
        groups = _build_groups_from_group_sizes(group_sizes)
    else:
        raise ValueError("groups_json or group_sizes required")
    _validate_worker_coverage(groups=groups, worker_count=worker_count)
    return RuntimeTopology(workers=workers, groups=groups)


def _default_group_json(args: argparse.Namespace) -> dict[str, Any]:
    tp = int(args.tp)
    ulysses = int(args.ulysses_degree)
    ring = int(args.ring_degree)
    cfg = int(args.cfg)
    size = tp * ulysses * ring * cfg
    if args.world_size is not None and int(args.world_size) != size:
        raise ValueError(
            "implicit single-group topology requires --world-size == --tp * "
            f"--ulysses-degree * --ring-degree * --cfg ({int(args.world_size)} != {size}); "
            "use --groups-json for custom or overlapping groups"
        )
    return {
        "group_id": "g0",
        "ranks": list(range(size)),
        "tp": tp,
        "ulysses_degree": ulysses,
        "ring_degree": ring,
        "cfg": cfg,
    }


def _worker_count(*, groups_json: list[dict[str, Any]] | None, group_sizes: str | None) -> int:
    if groups_json is not None:
        max_rank = -1
        cursor = 0
        for item in groups_json:
            if "ranks" in item:
                ranks = [int(rank) for rank in item["ranks"]]
                max_rank = max(max_rank, *ranks)
            else:
                size = int(item["size"])
                max_rank = max(max_rank, cursor + size - 1)
                cursor += size
        return max_rank + 1
    if group_sizes is not None:
        return sum(int(part.strip()) for part in group_sizes.split(",") if part.strip())
    raise ValueError("groups_json or group_sizes required")


def _build_groups_from_group_sizes(group_sizes: str) -> tuple[ExecutionGroupSpec, ...]:
    sizes = [int(part.strip()) for part in group_sizes.split(",") if part.strip()]
    if not sizes:
        raise ValueError("--group-sizes must not be empty")
    groups: list[ExecutionGroupSpec] = []
    cursor = 0
    for idx, size in enumerate(sizes):
        if size < 1:
            raise ValueError(f"group size must be positive, got {size}")
        ranks = tuple(range(cursor, cursor + size))
        cursor += size
        groups.append(
            ExecutionGroupSpec(
                group_id=f"g{idx}",
                ranks=ranks,
                parallel_spec=ParallelSpec(tp=size, sp=1, cfg=1),
                supported_task_kinds=STANDARD_TASK_KINDS,
                ulysses_degree=1,
                ring_degree=1,
            )
        )
    return tuple(groups)


def _build_groups_from_groups_json(groups_json: list[dict[str, Any]]) -> tuple[ExecutionGroupSpec, ...]:
    groups: list[ExecutionGroupSpec] = []
    next_rank = 0
    for idx, raw_group in enumerate(groups_json):
        group_id = str(raw_group.get("group_id", f"g{idx}"))
        if "ranks" in raw_group:
            ranks = tuple(int(rank) for rank in raw_group["ranks"])
            if not ranks:
                raise ValueError(f"group {group_id!r} must contain at least one rank")
            group_size = len(ranks)
        else:
            group_size = int(raw_group["size"])
            if group_size < 1:
                raise ValueError(f"group {group_id!r} size must be positive")
            ranks = tuple(range(next_rank, next_rank + group_size))
            next_rank += group_size

        tp = int(raw_group.get("tp", group_size))
        ulysses = int(raw_group.get("ulysses_degree", raw_group.get("ulysses", raw_group.get("sp", 1))))
        ring = int(raw_group.get("ring_degree", raw_group.get("ring", 1)))
        sp = ulysses * ring
        if "sp" in raw_group and int(raw_group["sp"]) != sp:
            raise ValueError(
                f"group {group_id!r} sequence parallel mismatch: "
                f"sp={int(raw_group['sp'])} != ulysses_degree({ulysses}) * ring_degree({ring})"
            )
        cfg = int(raw_group.get("cfg", 1))
        parallel_spec = ParallelSpec(tp=tp, sp=sp, cfg=cfg, cfg_parallel=bool(raw_group.get("cfg_parallel", False)))
        expected = parallel_spec.tp * parallel_spec.sp * parallel_spec.cfg
        if expected != len(ranks):
            raise ValueError(f"group {group_id!r} parallel mismatch: tp*sp*cfg={expected} != len(ranks)={len(ranks)}")
        groups.append(
            ExecutionGroupSpec(
                group_id=group_id,
                ranks=ranks,
                parallel_spec=parallel_spec,
                supported_task_kinds=_parse_supported_task_kinds(raw_group.get("supported_task_kinds")),
                ulysses_degree=ulysses,
                ring_degree=ring,
            )
        )
    if not groups:
        raise ValueError("--groups-json must contain at least one group")
    return tuple(groups)


def _parse_supported_task_kinds(raw: Any) -> tuple[TaskKind, ...]:
    if raw is None:
        return STANDARD_TASK_KINDS
    if not isinstance(raw, (list, tuple)):
        raise ValueError("supported_task_kinds must be a list")
    parsed = tuple(TaskKind(str(item)) for item in raw)
    if not parsed:
        raise ValueError("supported_task_kinds must not be empty")
    return parsed


def _validate_worker_coverage(*, groups: tuple[ExecutionGroupSpec, ...], worker_count: int) -> None:
    assigned: set[int] = set()
    for group in groups:
        assigned.update(group.ranks)
    expected = set(range(worker_count))
    missing = sorted(expected - assigned)
    if missing:
        raise ValueError(f"runtime_v2 groups do not cover all workers, missing ranks: {missing!r}")
    out_of_range = sorted(rank for rank in assigned if rank < 0 or rank >= worker_count)
    if out_of_range:
        raise ValueError(f"runtime_v2 groups contain out-of-range worker ranks: {out_of_range!r}")


def build_policy_factory(policy: str) -> Callable[[RuntimeTopology, TaskRuntimeEstimator], Any]:
    if ":" in policy:
        module_name, factory_name = policy.split(":", 1)
        module = importlib.import_module(module_name)
        factory = getattr(module, factory_name)
        return factory
    policy_name = policy.lower()
    if policy_name == "fcfs":
        return lambda topology, estimator: FCFSSchedulerPolicy(
            topology=topology,
            task_runtime_estimator=estimator,
        )
    if policy_name == "srtf":
        return lambda topology, estimator: SRTFSchedulerPolicy(
            topology=topology,
            task_runtime_estimator=estimator,
        )
    raise ValueError(f"unknown policy {policy!r}; use fcfs, srtf, or pkg.module:factory")


def load_workload(args: argparse.Namespace) -> list[SimRequest]:
    if args.workload_json:
        data = json.loads(Path(args.workload_json).read_text())
        return [_request_from_mapping(item) for item in data.get("requests", [])]
    return generate_synthetic_workload(args)


def _request_from_mapping(item: Mapping[str, Any]) -> SimRequest:
    deadline_ms = item.get("deadline_ms")
    profiled_optimal_latency_ms = item.get("profiled_optimal_latency_ms")
    preemptible = item.get("preemptible")
    do_true_cfg = item.get("do_true_cfg")
    latent_seq_len = item.get("latent_seq_len")
    voxels = item.get("voxels")
    if isinstance(do_true_cfg, str):
        do_true_cfg = do_true_cfg.strip().lower() in {"1", "true", "yes", "on"}
    return SimRequest(
        request_id=str(item["request_id"]),
        arrival_ms=float(item.get("arrival_ms", 0.0)),
        height=int(item.get("height", 720)),
        width=int(item.get("width", 1280)),
        num_frames=int(item.get("num_frames", 81)),
        num_steps=int(item.get("num_steps", 5)),
        denoise_chunk_size=int(item.get("denoise_chunk_size", 1)),
        text_seq_len=int(item.get("text_seq_len", 512)),
        priority=int(item.get("priority", 0)),
        group_id=item.get("group_id"),
        deadline_ms=float(deadline_ms) if deadline_ms is not None else None,
        profiled_optimal_latency_ms=(
            float(profiled_optimal_latency_ms) if profiled_optimal_latency_ms is not None else None
        ),
        request_class=item.get("request_class"),
        preemptible=bool(preemptible) if preemptible is not None else None,
        do_true_cfg=bool(do_true_cfg) if do_true_cfg is not None else None,
        latent_seq_len_override=int(latent_seq_len) if latent_seq_len is not None else None,
        voxels_override=int(voxels) if voxels is not None else None,
    )


def generate_synthetic_workload(args: argparse.Namespace) -> list[SimRequest]:
    rng = random.Random(args.seed)
    shapes = [_parse_shape(raw) for raw in args.shape_pool.split(",") if raw.strip()]
    requests: list[SimRequest] = []
    for idx in range(args.num_requests):
        if args.burst_size > 1:
            burst_idx = idx // args.burst_size
            in_burst_idx = idx % args.burst_size
            arrival = burst_idx * float(args.burst_gap_ms) + in_burst_idx * float(args.arrival_ms)
        else:
            arrival = idx * float(args.arrival_ms)
        height, width, frames = rng.choice(shapes)
        requests.append(
            SimRequest(
                request_id=f"r{idx}",
                arrival_ms=arrival,
                height=height,
                width=width,
                num_frames=frames,
                num_steps=args.num_steps,
                denoise_chunk_size=args.denoise_chunk_size,
                text_seq_len=args.text_seq_len,
                priority=0,
                group_id=args.group_id,
            )
        )
    return requests


def _parse_shape(raw: str) -> tuple[int, int, int]:
    parts = raw.lower().replace("*", "x").split("x")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"shape must look like HxWxF, got {raw!r}")
    return int(parts[0]), int(parts[1]), int(parts[2])


def build_chrome_trace_events(
    topology: RuntimeTopology,
    timeline: list[TimelineEvent],
    *,
    color_by: str = "kind",
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = [
        {
            "name": "process_name",
            "ph": "M",
            "pid": 1,
            "tid": 0,
            "args": {"name": "runtime_v2_policy_simulator"},
        },
        {
            "name": "process_sort_index",
            "ph": "M",
            "pid": 1,
            "tid": 0,
            "args": {"sort_index": 0},
        },
    ]
    for worker in topology.workers:
        tid = trace_tid(worker.worker_rank)
        events.append(
            {
                "name": "thread_name",
                "ph": "M",
                "pid": 1,
                "tid": tid,
                "args": {"name": f"rank {worker.worker_rank}"},
            }
        )
        events.append(
            {
                "name": "thread_sort_index",
                "ph": "M",
                "pid": 1,
                "tid": tid,
                "args": {"sort_index": worker.worker_rank},
            }
        )
    for item in timeline:
        if item.kind == SIM_TASK_LAUNCH_KIND:
            continue
        for rank in item.ranks:
            events.append(
                {
                    "name": f"{item.kind}[{item.request_id}]",
                    "cat": item.kind,
                    "cname": _trace_cname(item, color_by=color_by),
                    "ph": "X",
                    "pid": 1,
                    "tid": trace_tid(rank),
                    "ts": item.start_ms * 1000.0,
                    "dur": max(0.0, item.end_ms - item.start_ms) * 1000.0,
                    "args": item.args,
                }
            )
    return events


def trace_tid(rank: int) -> int:
    return int(rank) + 1


def _trace_cname(item: TimelineEvent, *, color_by: str) -> str:
    if color_by == "request":
        palette = (
            "good",
            "bad",
            "terrible",
            "rail_response",
            "rail_animation",
            "rail_load",
            "thread_state_iowait",
            "generic_work",
        )
        return palette[sum(ord(ch) for ch in item.request_id) % len(palette)]
    return _trace_cname_for_kind(item.kind)


def _trace_cname_for_kind(kind: str) -> str:
    return {
        TaskKind.TEXT_ENCODE.value: "rail_response",
        TaskKind.DIT_PREPARE.value: "rail_animation",
        TaskKind.TIMESTEP_PREPARE.value: "rail_load",
        TaskKind.DIT_STEP_CHUNK.value: "good",
        TaskKind.RESHARD.value: "thread_state_iowait",
        TaskKind.VAE_DECODE.value: "bad",
        TaskKind.FINALIZE.value: "terrible",
    }.get(str(kind), "generic_work")


def compute_global_idle(
    timeline: list[TimelineEvent],
    *,
    start_ms: float,
    end_ms: float,
) -> tuple[float, float]:
    if end_ms <= start_ms:
        return 0.0, 0.0
    intervals = sorted((event.start_ms, event.end_ms) for event in timeline if event.end_ms > event.start_ms)
    if not intervals:
        return end_ms - start_ms, 1.0
    busy = 0.0
    cur_start, cur_end = intervals[0]
    cur_start = max(cur_start, start_ms)
    cur_end = min(cur_end, end_ms)
    for start, end in intervals[1:]:
        start = max(start, start_ms)
        end = min(end, end_ms)
        if end <= start:
            continue
        if start <= cur_end:
            cur_end = max(cur_end, end)
            continue
        busy += max(0.0, cur_end - cur_start)
        cur_start, cur_end = start, end
    busy += max(0.0, cur_end - cur_start)
    total = end_ms - start_ms
    idle = max(0.0, total - busy)
    return idle, idle / total if total > 0 else 0.0


def write_trace(path: str | Path, events: list[dict[str, Any]]) -> None:
    Path(path).write_text(json.dumps({"traceEvents": events}, indent=2, sort_keys=True))


def print_summary(result: SimulationResult) -> None:
    latencies = [
        result.request_finish_ms[request_id] - arrival
        for request_id, arrival in result.request_arrival_ms.items()
        if request_id in result.request_finish_ms
    ]
    makespan = max(result.request_finish_ms.values(), default=0.0)
    print(f"completed_requests={len(latencies)}")
    print(f"makespan_ms={makespan:.3f}")
    print(f"global_idle_ms={result.global_idle_ms:.3f}")
    print(f"global_idle_frac={result.global_idle_frac:.6f}")
    if not latencies:
        return
    print(f"latency_avg_ms={statistics.mean(latencies):.3f}")
    print(f"latency_p50_ms={statistics.median(latencies):.3f}")
    print(f"latency_max_ms={max(latencies):.3f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cost-model-dir", default="cost-model/wan22-ti2v-5b")
    parser.add_argument("--policy", default="srtf", help="fcfs, srtf, or pkg.module:factory")
    parser.add_argument("--trace-out", default="runtime_v2_policy_sim_trace.json")
    parser.add_argument(
        "--groups-json",
        default=None,
        help="explicit topology; each group may set ranks/tp/ulysses_degree/ring_degree/cfg",
    )
    parser.add_argument("--group-sizes", default=None, help="legacy partition shorthand; groups use tp=size, sp=1")
    parser.add_argument(
        "--world-size",
        type=int,
        default=None,
        help="physical rank count; default is --tp * --ulysses-degree * --ring-degree * --cfg",
    )
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--ulysses-degree", type=int, default=1)
    parser.add_argument("--ring-degree", type=int, default=1)
    parser.add_argument("--cfg", type=int, default=1)
    parser.add_argument("--reshard-ms", type=float, default=0.0)
    parser.add_argument(
        "--launch-end-before-exec-end-ms",
        "--launch-overhead-ms",
        dest="launch_end_before_exec_end_ms",
        type=float,
        default=0.05,
        help=(
            "simulated gap between launch end and exec end; "
            "--launch-overhead-ms is kept as a compatibility alias"
        ),
    )
    parser.add_argument(
        "--exec-start-delay-ms",
        type=float,
        default=0.005,
        help="delay from launch begin to earliest GPU execution begin",
    )
    parser.add_argument(
        "--round-ms",
        type=float,
        default=0.0,
        help=(
            "fixed round length (ms) for round-based policies (e.g. "
            "make_tetriserve_round); 0 disables round ticks"
        ),
    )

    parser.add_argument("--workload-json", default=None)
    parser.add_argument("--num-requests", type=int, default=8)
    parser.add_argument(
        "--arrival-ms",
        type=float,
        default=0.0,
        help="request interarrival interval; in burst mode, interval within each burst",
    )
    parser.add_argument("--burst-size", type=int, default=1, help="when >1, enable burst arrival mode")
    parser.add_argument("--burst-gap-ms", type=float, default=0.0, help="time between burst starts")
    parser.add_argument("--shape-pool", default="480x832x17,720x1280x81")
    parser.add_argument("--num-steps", type=int, default=5)
    parser.add_argument("--denoise-chunk-size", type=int, default=1)
    parser.add_argument("--text-seq-len", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--group-id", default=None, help="optional explicit group_id for all synthetic requests")
    parser.add_argument("--debug-scheduler", action="store_true")
    parser.add_argument("--trace-color-by", choices=("kind", "request"), default="kind")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cost_models = CostModelStore.load_dir(args.cost_model_dir)
    topology = build_topology(args, cost_models)
    requests = load_workload(args)
    simulator = RuntimeV2PolicySimulator(
        topology=topology,
        policy_factory=build_policy_factory(args.policy),
        cost_models=cost_models,
        reshard_ms=args.reshard_ms,
        launch_end_before_exec_end_ms=args.launch_end_before_exec_end_ms,
        exec_start_delay_ms=args.exec_start_delay_ms,
        round_ms=args.round_ms,
        trace_color_by=args.trace_color_by,
        debug_scheduler=args.debug_scheduler,
    )
    result = simulator.run(requests)
    write_trace(args.trace_out, result.trace_events)
    print_summary(result)
    print(f"wrote_trace={args.trace_out}")


if __name__ == "__main__":
    main()
