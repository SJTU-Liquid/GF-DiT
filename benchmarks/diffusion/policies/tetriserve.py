# SPDX-License-Identifier: Apache-2.0
"""TetriServe-style step-level SP scheduling for the runtime_v2 simulator.

Implements the two-stage scheduler of TetriServe (arXiv 2510.01565):

  1. Deadline-aware minimal GPU allocation -- for each active request, find
     the smallest SP degree whose per-step cost still lets the request finish
     by its deadline (minimizing GPU-hours k*T(k)).
  2. Group-knapsack "survival" packing (paper Algorithm 1) -- over the FULL
     N-GPU pool and ALL active requests (running + pending), pick at most one
     allocation per request, total GPUs <= N, maximizing the number of
     requests that remain *not-definitely-late*.

Two variants share this brain and differ only in *when* they re-plan:

  * ``TetriServeContinuousPolicy`` (factory ``make_tetriserve_continuous``)
    re-plans on every chunk-completion / arrival event. Because ranks free
    up asynchronously (a request finishes its chunk one rank at a time), it
    cannot assemble a wide SP group from a single event's free ranks. It
    therefore *reserves* ranks: when the global plan wants SP=k for an urgent
    request but only j<k ranks are free, it holds those j ranks idle (blocking
    lower-priority requests from grabbing them) and waits for the in-flight
    chunks the plan chose to drain to reach their step boundary. No mid-chunk
    preemption. The drain idle is bounded by one chunk, far smaller than a
    fixed round.

  * ``TetriServeRoundPolicy`` (in ``tetriserve_round.py``) re-plans only at
    fixed round boundaries -- faithful to the paper, and the baseline whose
    round-boundary fragmentation this continuous variant is meant to beat.

The deadline-aware allocation + dynamic group synthesis + placement
preservation are reused from EdfBestFitPolicy / EdfGreedyPolicy.
"""

from __future__ import annotations

import math
import os
from typing import Iterable

from vllm_omni.diffusion.runtime_v2.protocol import (
    InferenceTask,
    ParallelSpec,
    RequestExecutionPlan,
    TaskKind,
    WorkerEvent,
    WorkerEventKind,
)
from vllm_omni.diffusion.runtime_v2.scheduler import TaskRuntimeEstimator
from vllm_omni.diffusion.runtime_v2.topology import RuntimeTopology

from benchmarks.diffusion.policies.edf_best_fit import (
    EdfBestFitPolicy,
    _EdfBestFitRequestMeta,
)


class TetriServeContinuousPolicy(EdfBestFitPolicy):
    """Continuous (event-driven) TetriServe with rank reservation/drain."""

    _DIT_TASK_KINDS = (
        TaskKind.DIT_PREPARE,
        TaskKind.TIMESTEP_PREPARE,
        TaskKind.DIT_STEP_CHUNK,
    )

    def __init__(
        self,
        topology: RuntimeTopology,
        task_runtime_estimator: TaskRuntimeEstimator | None = None,
        *,
        allowed_sp_sizes: Iterable[int] | None = None,
        elastic_scale_up: bool = True,
    ) -> None:
        super().__init__(
            topology=topology,
            task_runtime_estimator=task_runtime_estimator,
            allowed_sp_sizes=allowed_sp_sizes,
        )
        self.elastic_scale_up = bool(elastic_scale_up)
        self.num_workers = max(1, len(topology.workers))
        # Per-request completed DiT steps and the in-flight task (for planning
        # the *next* allocation of a request that is mid-chunk).
        self.completed_steps: dict[str, int] = {}
        self._inflight_task: dict[str, InferenceTask] = {}

    # ---- bookkeeping overrides -------------------------------------------

    def _dispatch(self, task, request_id, group_id, ranks):
        super()._dispatch(task, request_id, group_id, ranks)
        self._inflight_task[request_id] = task

    def _release_dispatched_task(self, event: WorkerEvent):
        task = self._task_index.get(event.task_id)
        if (
            task is not None
            and task.kind == TaskKind.DIT_STEP_CHUNK
            and task.step_range is not None
        ):
            self.completed_steps[event.request_id] = max(
                self.completed_steps.get(event.request_id, 0),
                int(task.step_range.end),
            )
        self._inflight_task.pop(event.request_id, None)
        super()._release_dispatched_task(event)

    def _drop_request(self, request_id: str) -> None:
        self.completed_steps.pop(request_id, None)
        self._inflight_task.pop(request_id, None)
        super()._drop_request(request_id)

    # ---- scheduling core --------------------------------------------------

    def _take_dispatchable_tasks(self):
        # Fall back to EdfBestFit's greedy best-fit when there is no cost model
        # (no deadline math possible -> survival DP is meaningless).
        if self.task_runtime_estimator is None:
            return super()._take_dispatchable_tasks()

        free_ranks = self._free_rank_set()
        out: list[InferenceTask] = []

        ready_dit: list[tuple[str, InferenceTask]] = []
        ready_aux: list[tuple[str, InferenceTask]] = []
        for request_id, queue in self.ready_by_request.items():
            if not queue or self.request_inflight.get(request_id, 0) > 0:
                continue
            task = queue[0]
            if task.group_id is not None:
                # Pinned task fits one way; materialize before the survival DP.
                placement = self._select_placement(task, free_ranks)
                if placement is not None:
                    group_id, ranks = placement
                    self._dispatch(task, request_id, group_id, ranks)
                    free_ranks.difference_update(ranks)
                    out.append(task)
            elif task.kind in self._SINGLE_RANK_TASK_KINDS:
                ready_aux.append((request_id, task))
            elif task.kind in self._DIT_TASK_KINDS:
                ready_dit.append((request_id, task))

        # Aux stages (text_encode/VAE/finalize) are single-rank and short, and
        # in the paper's deployment they run on a separate small encoder GPU off
        # the DiT critical path. We mirror that intent by dispatching ready aux
        # tasks FIRST on any free rank (EDF order) -- BEFORE the DiT planner
        # runs. This stops the prior pathology where the DiT reservation would
        # grab all 4 ranks for an SP4 group, leaving no rank for aux, hiding
        # the queued-aux requests from `_active_dit_request_ids`, and tricking
        # scale_up / `_min_survive_sp` into a lone-request SP4 cycle.
        ready_aux.sort(key=lambda rt: self._sort_key(rt[0]))
        for request_id, task in ready_aux:
            if not free_ranks:
                break
            rank = (min(free_ranks),)
            group = self._ensure_group_for_ranks(
                ranks=rank, supported_task_kinds=self._STANDARD_TASK_KINDS
            )
            self._dispatch(task, request_id, group.group_id, rank)
            free_ranks.difference_update(rank)
            out.append(task)

        # TetriServe admission (Algorithm 1 survival knapsack) over the full
        # N-GPU pool and ALL active DiT requests, run continuously (every event).
        target_sp = self._plan_target_allocation()

        # Realize the plan in EDF urgency order. A request whose target width
        # cannot be assembled from currently-free ranks reserves (holds idle)
        # the free ranks it would use, blocking less-urgent requests, and waits
        # for in-flight chunks to drain to their step boundary (async assembly).
        held: set[int] = set()
        ready_dit.sort(key=lambda rt: self._sort_key(rt[0]))
        for request_id, task in ready_dit:
            sp = int(target_sp.get(request_id, 0))
            if sp <= 0:
                continue  # did not fit the budget this round
            avail = free_ranks - held
            chosen = self._pick_ranks_for_request(request_id, sp, avail)
            if len(chosen) >= sp and set(chosen).issubset(avail):
                ranks = tuple(chosen[:sp])
                group = self._ensure_group_for_ranks(
                    ranks=ranks, supported_task_kinds=self._DIT_TASK_KINDS
                )
                self._dispatch(task, request_id, group.group_id, ranks)
                free_ranks.difference_update(ranks)
                out.append(task)
            else:
                held.update(sorted(avail)[:sp])  # reserve + idle, wait for drain

        return out

    # ---- TetriServe planner ----------------------------------------------

    def _plan_target_allocation(self) -> dict[str, int]:
        """TetriServe admission (paper Algorithm 1), shared by the continuous (A)
        and round (B) variants -- they differ ONLY in the planning window
        (`_planning_window_ms`: one chunk for async A, one round for B) and in
        WHEN this runs (every event for A, every round tick for B).

        Per active request we build the option set (`_survival_options`): a
        width-0 "none" option carrying its survives-if-deferred bit, and a
        width-k* "run" option at its deadline-aware minimal SP. The group-knapsack
        (`_survival_knapsack`) maximizes the number of survivors under the N-GPU
        budget; leftover GPUs go to un-admitted requests (work-conserving fill +
        already-late best-effort) and then elastic scale-up.
        """
        active = self._active_dit_request_ids()
        options_by_request: list[tuple[str, list[tuple[int, float]]]] = []
        for request_id in active:
            opts = self._survival_options(request_id)
            if opts:
                options_by_request.append((request_id, opts))
        target = self._survival_knapsack(options_by_request, self.num_workers)

        # Work-conserving fill: leftover GPUs -> un-admitted active requests
        # (slack survivors the DP deferred + already-late best-effort at SP1).
        budget = self.num_workers - sum(target.values())
        for request_id in sorted(active, key=self._sort_key):
            if budget < 1:
                break
            if request_id in target:
                continue
            sp, _ = self._min_survive_sp(request_id)
            if sp <= budget:
                target[request_id] = sp
                budget -= sp
            else:
                target[request_id] = 1  # best-effort single GPU
                budget -= 1
        if self.elastic_scale_up:
            self._scale_up(target)
        return target

    def _planning_window_ms(self, meta, t_min: float) -> float:
        """How long a request waits before the scheduler reconsiders it if not
        run now -- the cost of deferring it. Continuous A re-plans on every
        event, so the window is ~one chunk at the fastest SP. (Round B overrides
        this to the fixed round length.)"""
        return max(1, int(meta.chunk_steps)) * t_min

    def _survival_options(self, request_id: str) -> list[tuple[int, float]]:
        """Algorithm 1 option set for one request:
          - none: (width 0, value = survives-if-deferred-one-window)
          - run:  (width k*, value 1) where k* is the deadline-aware minimal SP.
        Survival-if-deferred LB: now + window + remaining*T_min <= deadline.
        """
        meta = self.request_meta.get(request_id)
        if meta is None:
            return []
        remaining = self._remaining_steps(request_id)
        if remaining <= 0:
            return []
        candidates = sorted(s for s in self.allowed_sp_sizes if s <= self.num_workers)
        if not candidates:
            return []
        t_min = min(
            (
                self._safe_estimate(
                    ParallelSpec(tp=1, sp=k, cfg=1), meta.latent_seq_len, "dit_step_chunk"
                )
                for k in candidates
            ),
            default=0.0,
        )
        deadline = meta.deadline_ms
        if deadline is None or not math.isfinite(deadline):
            sv_none = 1.0
        else:
            window = self._planning_window_ms(meta, t_min)
            deferred_finish = self.now_ms + window + remaining * t_min
            sv_none = 1.0 if deferred_finish <= deadline else 0.0
        options: list[tuple[int, float]] = [(0, sv_none)]
        run_sp, can_survive = self._min_survive_sp(request_id)
        if can_survive:
            options.append((run_sp, 1.0 - 1e-6 * run_sp))
        return options

    def _min_survive_sp(self, request_id: str) -> tuple[int, bool]:
        """Deadline-aware minimal allocation: (smallest SP that finishes by the
        deadline, True), or (widest SP, False) when no SP can meet it. No
        deadline -> (smallest SP, True)."""
        meta = self.request_meta.get(request_id)
        candidates = sorted(s for s in self.allowed_sp_sizes if s <= self.num_workers)
        if meta is None or not candidates:
            return (1, True)
        deadline = meta.deadline_ms
        if deadline is None or not math.isfinite(deadline):
            return (candidates[0], True)
        for sp in candidates:
            if self._estimate_finish_ms(request_id, sp) <= deadline:
                return (sp, True)
        return (candidates[-1], False)

    def _active_dit_request_ids(self) -> list[str]:
        ids: list[str] = []
        for request_id, meta in self.request_meta.items():
            if meta.num_steps <= 0 or self._remaining_steps(request_id) <= 0:
                continue
            if self.request_inflight.get(request_id, 0) > 0:
                # In-flight: only counts as an active DiT request if the chunk it
                # is running is a DiT chunk (aux stages do not reserve DiT ranks).
                task = self._inflight_task.get(request_id)
                if task is not None and task.kind in self._DIT_TASK_KINDS:
                    ids.append(request_id)
                continue
            queue = self.ready_by_request.get(request_id)
            if queue and queue[0].kind in self._DIT_TASK_KINDS:
                ids.append(request_id)
        return ids

    def _survival_knapsack(
        self,
        options_by_request: list[tuple[str, list[tuple[int, float]]]],
        capacity: int,
    ) -> dict[str, int]:
        """Group-knapsack (paper Algorithm 1): pick exactly one option per
        request, total width <= capacity, maximize summed survival value.
        Returns {request_id: sp} for requests whose chosen option has width > 0.
        """
        # dp[c] = (best_value, {request_id: chosen_width}) using at most c GPUs.
        dp: list[tuple[float, dict[str, int]]] = [
            (-math.inf, {}) for _ in range(capacity + 1)
        ]
        dp[0] = (0.0, {})
        for request_id, options in options_by_request:
            nxt: list[tuple[float, dict[str, int]]] = [
                (-math.inf, {}) for _ in range(capacity + 1)
            ]
            for c in range(capacity + 1):
                base_value, base_pick = dp[c]
                if base_value == -math.inf:
                    continue
                for width, value in options:
                    nc = c + width
                    if nc > capacity:
                        continue
                    cand_value = base_value + value
                    if cand_value > nxt[nc][0]:
                        pick = dict(base_pick)
                        if width > 0:
                            pick[request_id] = width
                        nxt[nc] = (cand_value, pick)
            dp = nxt
        best = max(dp, key=lambda cell: cell[0])
        return dict(best[1])

    def _scale_up(self, target: dict[str, int]) -> None:
        """Work-conserving elastic scale-up: hand leftover GPU budget to the
        most-urgent admitted requests, promoting them to the next allowed SP.
        """
        used = sum(target.values())
        budget = self.num_workers - used
        if budget <= 0:
            return
        sorted_allowed_asc = sorted(self.allowed_sp_sizes)
        order = sorted(target.keys(), key=self._sort_key)
        grew = True
        while budget > 0 and grew:
            grew = False
            for request_id in order:
                cur = target[request_id]
                bigger = next((s for s in sorted_allowed_asc if s > cur), None)
                if bigger is None:
                    continue
                delta = bigger - cur
                if delta > budget:
                    continue
                # Only promote if more parallelism is actually faster for this
                # request (sublinear scaling can make wider SP a regression).
                if not self._promotion_helps(request_id, cur, bigger):
                    continue
                target[request_id] = bigger
                budget -= delta
                grew = True
                break

    def _promotion_helps(self, request_id: str, cur_sp: int, new_sp: int) -> bool:
        meta = self.request_meta.get(request_id)
        if meta is None:
            return False
        cur = self._safe_estimate(
            ParallelSpec(tp=1, sp=cur_sp, cfg=1), meta.latent_seq_len, "dit_step_chunk"
        )
        new = self._safe_estimate(
            ParallelSpec(tp=1, sp=new_sp, cfg=1), meta.latent_seq_len, "dit_step_chunk"
        )
        return new < cur

    def _remaining_steps(self, request_id: str) -> int:
        meta = self.request_meta.get(request_id)
        if meta is None:
            return 0
        completed = self.completed_steps.get(request_id, 0)
        # An in-flight chunk is counted as done for *next-allocation* planning.
        task = self._inflight_task.get(request_id)
        if task is not None and task.step_range is not None:
            completed = max(completed, int(task.step_range.end))
        else:
            queue = self.ready_by_request.get(request_id)
            if queue and queue[0].step_range is not None:
                completed = max(completed, int(queue[0].step_range.start))
        return max(0, meta.num_steps - completed)


class TetriServeRoundPolicy(TetriServeContinuousPolicy):
    """Faithful round-based TetriServe (Route B): re-plan ONLY at fixed round
    boundaries.

    Admission and SP allocation happen exclusively on ``on_round_tick`` (driven
    by the simulator's fixed round_ms ticks). Within a round, an already-admitted
    DiT request continues its next chunk back-to-back on the SAME group it was
    admitted to, but NOTHING new is admitted between ticks: a request arriving
    mid-round waits for the next boundary, an aux->DiT transition waits for the
    next boundary, and ranks freed by a request that finishes (or was not
    admitted) sit idle until the next boundary. That round-boundary fragmentation
    is the structural waste this variant exists to expose -- it should trail the
    continuous variant, which reclaims those gaps immediately.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._gate_open = False
        self._round_end_ms = 0.0
        # Round length; the simulator sets this to its --round-ms before run().
        self.round_ms = 0.0

    def _take_dispatchable_tasks(self):
        # Dispatch only when a round boundary has opened the gate.
        if not self._gate_open:
            return []
        return super()._take_dispatchable_tasks()

    def on_round_tick(self, now_ms: float):
        self.now_ms = max(self.now_ms, float(now_ms))
        self._round_end_ms = self.now_ms + self.round_ms
        self._gate_open = True
        try:
            return self._take_dispatchable_tasks()
        finally:
            self._gate_open = False

    def on_worker_event(self, event: WorkerEvent):
        # super() does the release/drop bookkeeping; its gated _take_dispatchable
        # call returns nothing. Then continue any admitted request within-round.
        dispatched = list(super().on_worker_event(event))
        dispatched.extend(self._continue_within_round(event))
        return dispatched

    def _continue_within_round(self, event: WorkerEvent) -> list[InferenceTask]:
        if event.kind not in (WorkerEventKind.TASK_EXEC_END,):
            return []
        request_id = event.request_id
        if self.request_inflight.get(request_id, 0) > 0:
            return []
        queue = self.ready_by_request.get(request_id)
        if not queue:
            return []
        task = queue[0]
        # Continue only a request that is ALREADY mid-DiT-run this round: once it
        # was admitted to a DiT group at a boundary, let its DiT sub-stages
        # (DIT_PREPARE -> TIMESTEP_PREPARE -> DIT_STEP_CHUNK) flow back-to-back on
        # that same group within the round. The transition INTO the DiT run
        # (TEXT_ENCODE -> DIT_PREPARE) and trailing aux (VAE/FINALIZE) still wait
        # for a tick -- the former so the round planner assigns the SP group, the
        # latter because they are single-rank stages. "Mid-DiT-run" = the task
        # that just finished was itself a DiT task.
        completed = self._task_index.get(event.task_id)
        if completed is None or completed.kind not in self._DIT_TASK_KINDS:
            return []
        if task.kind not in self._DIT_TASK_KINDS:
            return []
        last_group_id = self.request_last_group.get(request_id)
        if last_group_id is None:
            return []
        try:
            group = self.topology.get_group(last_group_id)
        except (KeyError, AttributeError):
            return []
        ranks = tuple(int(rank) for rank in group.ranks)
        if int(group.parallel_spec.sp) != len(ranks):
            return []  # not a clean DiT group
        free = self._free_rank_set()
        if not set(ranks).issubset(free):
            return []  # its ranks are not free -> would need a re-plan; wait
        # Round barrier: only continue if this sub-stage finishes before the
        # round boundary. Otherwise pause -> the request's ranks free up and sit
        # idle until the next tick re-plans over a clean, fully-free pool. This
        # is the structural round-boundary waste (tail idle + admission latency).
        if self.now_ms + self._estimate_task_ms(task, last_group_id) > self._round_end_ms + 1e-6:
            return []
        self._dispatch(task, request_id, last_group_id, ranks)
        return [task]

    def _estimate_task_ms(self, task: InferenceTask, group_id: str) -> float:
        """Estimated wall time of one task on group_id, using its own cost-model
        stage (DIT_PREPARE/TIMESTEP_PREPARE are one-shots; DIT_STEP_CHUNK scales
        by its step count)."""
        if self.task_runtime_estimator is None:
            return 0.0
        sp = int(self.topology.get_group(group_id).parallel_spec.sp)
        seq_len = int(task.payload.get("seq_len", 1))
        stage = str(task.payload.get("cost_model_stage") or task.kind.value)
        est = self._safe_estimate(ParallelSpec(tp=1, sp=sp, cfg=1), seq_len, stage)
        if task.kind == TaskKind.DIT_STEP_CHUNK and task.step_range is not None:
            est *= max(1, task.step_range.end - task.step_range.start)
        return est

    # Round B uses the SAME TetriServe admission (Algorithm 1 survival knapsack)
    # as the continuous variant; it only widens the planning/defer window to a
    # full round (vs A's one-chunk) and runs it at fixed boundaries.
    def _planning_window_ms(self, meta, t_min: float) -> float:
        return self.round_ms


def make_tetriserve_continuous(topology, task_runtime_estimator):
    raw = os.getenv("EDF_GREEDY_SP_SIZES", "").strip()
    allowed_sp_sizes: tuple[int, ...] | None = None
    if raw:
        allowed_sp_sizes = tuple(int(s.strip()) for s in raw.split(",") if s.strip())
    return TetriServeContinuousPolicy(
        topology=topology,
        task_runtime_estimator=task_runtime_estimator,
        allowed_sp_sizes=allowed_sp_sizes,
        elastic_scale_up=os.getenv("TETRISERVE_ELASTIC_SCALE_UP", "1") != "0",
    )


def make_tetriserve_round(topology, task_runtime_estimator):
    raw = os.getenv("EDF_GREEDY_SP_SIZES", "").strip()
    allowed_sp_sizes: tuple[int, ...] | None = None
    if raw:
        allowed_sp_sizes = tuple(int(s.strip()) for s in raw.split(",") if s.strip())
    return TetriServeRoundPolicy(
        topology=topology,
        task_runtime_estimator=task_runtime_estimator,
        allowed_sp_sizes=allowed_sp_sizes,
        elastic_scale_up=os.getenv("TETRISERVE_ELASTIC_SCALE_UP", "1") != "0",
    )
