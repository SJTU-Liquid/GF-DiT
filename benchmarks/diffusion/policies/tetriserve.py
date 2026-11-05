# SPDX-License-Identifier: Apache-2.0
"""TetriServe-style step-level SP scheduling for the runtime_v2 simulator.

This module holds three policies:

  * ``TetriServeRoundPolicy`` (factory ``make_tetriserve_round``) -- a direct
    port of the *released* TetriServe SLO scheduler
    (``tetriserve/scheduler/dyn_step_scheduler.py::dyn_slo_schedule_global``,
    ASPLOS'26 artifact). It reproduces, step for step: per-request optimal
    shape search (``get_optimal_shape_python``), the step-granularity-derived
    round duration (``_calculate_round_duration``), the DP round scheduler
    (``_dynamic_programming_combo_search``), bubble-filling scale-up
    (``_scale_up_ranks_to_fill_bubble``), placement-preserving rank assignment
    (``_assign_ranks_to_combos``), step synchronization
    (``_adjust_steps_for_synchronization``) and the step-granularity block
    chopping of ``_build_schedule_DAG``. See ``ALIGNMENT_REPORT.md`` (scratch)
    and the per-method docstrings below for the residual divergences.

  * ``TetriServeFixedRoundPolicy`` (factory ``make_tetriserve_round_fixed``) --
    the PREVIOUS, unfaithful round variant: a fixed wall-clock round barrier
    of ``--round-ms`` with no step-granularity round duration and no step
    synchronization. Kept only so the alignment report can quantify what the
    old approximation cost. Do not use it as "the TetriServe baseline".

  * ``TetriServeContinuousPolicy`` (factory ``make_tetriserve_continuous``) --
    OUR event-driven variant. It keeps TetriServe's deadline-aware minimal
    allocation + survival knapsack but re-plans on every chunk-completion /
    arrival event and reserves ranks across asynchronous drains. This is not a
    port; it is the contrast case.

Notes on the released scheduler that drive the port's structure:

  * TetriServe's ``schedule()`` is invoked from ``_CServeEngine.schedule_tasks``
    on EVERY task completion and on every arrival, over the currently-FREE rank
    pool (``Scheduler.ranks`` shrinks on dispatch, grows on completion). There
    is no wall-clock round barrier anywhere in the released code. The "round"
    is a *work-commit horizon*: a planning call decides how many denoising
    steps each admitted request executes before the scheduler reconsiders it.
  * The DAG built at the end of a planning call is *replaced* wholesale on the
    next call (``schedule_cache.update_schedule_DAG``), and blocks of one combo
    are chained on their shared ranks, so only the FIRST block of each combo
    (at most ``_get_step_granularity`` steps) is ever executed before the next
    re-plan. The port therefore commits
    ``min(sync_truncated_steps, _get_step_granularity(pixels))`` steps.
  * ``greedy_bin_packing_exhaustive`` ``break``s after one iteration, so
    ``rank_time`` is all-zero: step synchronization truncates every combo to
    the *shortest* admitted combo's duration.
"""

from __future__ import annotations

import bisect
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

# ---------------------------------------------------------------------------
# Constants lifted verbatim from the released TetriServe scheduler.
# ---------------------------------------------------------------------------

# dyn_step_scheduler.py:31 -- "7% tolerance for SLO compliance:
# latency <= target_slo * 1.07". The released scheduler uses this inside its
# survival predicate AND its reported SLO metric. Our runner's metric has NO
# tolerance (`finish <= deadline_ms`), so keeping 0.07 makes the ported
# scheduler slightly optimistic about its own deadlines; `make_tetriserve_round`
# exposes TETRISERVE_SLO_TOLERANCE for the sensitivity check reported in
# ALIGNMENT_REPORT.md.
SLO_TOLERANCE = 0.07

# The GPU configurations TetriServe enumerates everywhere (1/2/4/8 GPUs).
_GPU_COUNTS = (1, 2, 4, 8)


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


class TetriServeFixedRoundPolicy(TetriServeContinuousPolicy):
    """DEPRECATED, kept for the alignment report only.

    The PREVIOUS (unfaithful) round variant: re-plan ONLY at fixed wall-clock
    round boundaries of ``--round-ms``. The released TetriServe has no such
    barrier -- it derives a round *duration* from step granularity and
    synchronizes combos to remove the end-of-round bubble. Use
    ``TetriServeRoundPolicy`` for anything that claims to be TetriServe.

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


class _TetriCombo:
    """One entry of TetriServe's ``best_combo_list``: (req_idx, rank_group,
    req_step, req_time). ``gi`` is the log2 GPU-config index into shape_state."""

    __slots__ = ("idx", "gpus", "gi", "steps")

    def __init__(self, idx: int, gpus: int, gi: int, steps: int) -> None:
        self.idx = idx
        self.gpus = gpus
        self.gi = gi
        self.steps = steps


class _TetriCandidate:
    """One entry of TetriServe's ``unscheduled_reqs`` plus its ``shape_state``
    row (steps allocated to the 1/2/4/8-GPU configs)."""

    __slots__ = ("request_id", "task", "meta", "pixels", "shape", "remaining")

    def __init__(self, request_id, task, meta, pixels, shape, remaining) -> None:
        self.request_id = request_id
        self.task = task
        self.meta = meta
        self.pixels = pixels
        self.shape = shape
        self.remaining = remaining


class TetriServeRoundPolicy(TetriServeContinuousPolicy):
    """Aligned port of the released TetriServe SLO scheduler.

    Maps ``dyn_slo_schedule_global()`` onto our task-graph interface. One call
    to :meth:`_take_dispatchable_tasks` == one call to their ``schedule()``:
    it re-plans over the currently-FREE rank pool, exactly like theirs (their
    ``self.ranks`` is the free-rank list, and ``schedule_tasks()`` runs on every
    task completion and arrival). The ordering of the planning stages mirrors
    ``dyn_slo_schedule_global`` line for line:

      1. ``_filter_and_prepare_requests``  -> :meth:`_candidates`
         (drop running requests, FCFS order, truncate to the free-GPU count)
      2. ``_compute_optimal_shapes`` / ``get_optimal_shape_python``
                                          -> :meth:`_optimal_shape`
      3. ``_calculate_round_duration``     -> :meth:`_round_duration_ms`
      4. ``_dynamic_programming_combo_search`` -> :meth:`_dp_combo_search`
      5. ``_scale_up_ranks_to_fill_bubble``-> :meth:`_scale_up_to_fill_bubble`
      6. ``_assign_ranks_to_combos``       -> :meth:`_assign_ranks`
      7. ``_adjust_steps_for_synchronization`` -> :meth:`_synchronize_steps`
      8. ``_build_schedule_DAG`` block chopping -> the
         ``min(steps, _get_step_granularity(pixels))`` commit cap

    Because their engine hands a whole (req, ranks, steps) block to a runner as
    ONE unit while our plan is one task per denoising step, a combo becomes a
    *commitment*: the request runs exactly ``steps`` ``DIT_STEP_CHUNK`` tasks
    back-to-back on the assigned ranks and is invisible to the planner until the
    commitment drains (mirroring ``running_requests`` exclusion in theirs).

    There is no wall-clock round barrier: the round *duration* is an internal
    horizon computed from step granularity, never a gate. ``--round-ms`` is
    therefore optional (``self_scheduled_rounds``); if supplied, its ticks are
    just extra harmless re-plans.
    """

    # Tells RuntimeV2PolicySimulator this policy derives its own round length
    # and does not need simulator-driven ticks, so --round-ms may be 0.
    self_scheduled_rounds = True

    def __init__(
        self,
        *args,
        gpu_is_h100: bool = False,
        slo_tolerance: float = SLO_TOLERANCE,
        placement_preserving: bool = True,
        admission_order: str = "fcfs",
        candidate_limit: str = "free",
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        # get_gpu_name() == "H100_HBM3" selects a different granularity /
        # MIN_STEPS_PER_CONFIG / scale-up branch in the released code. Our cost
        # model is H20, so the non-H100 branch is the faithful one.
        self.gpu_is_h100 = bool(gpu_is_h100)
        self.slo_tolerance = float(slo_tolerance)
        self.placement_preserving = bool(placement_preserving)
        # Sensitivity knobs for the alignment report. The faithful settings are
        # the defaults; they are NOT tuned. "edf" / "all" exist only to show
        # whether the gap vs our EDF policy is an artifact of TetriServe's FCFS
        # admission order and its truncation of the candidate list.
        if admission_order not in {"fcfs", "edf"}:
            raise ValueError(f"admission_order must be fcfs|edf, got {admission_order!r}")
        if candidate_limit not in {"free", "all"}:
            raise ValueError(f"candidate_limit must be free|all, got {candidate_limit!r}")
        self.admission_order = admission_order
        self.candidate_limit = candidate_limit
        # Unused by the algorithm; kept so the simulator's round-tick plumbing
        # can still set it without AttributeError.
        self.round_ms = 0.0
        self._pixels: dict[str, int] = {}
        # Live work commitments (a combo that has not finished its steps yet).
        self._commit_steps: dict[str, int] = {}
        self._commit_ranks: dict[str, tuple[int, ...]] = {}
        self._frontier_cache: dict[tuple[int, int, int], tuple] = {}
        self._tstep_cache: dict[tuple[int, int], float] = {}
        # Diagnostics for the alignment report (never used for decisions).
        self.round_duration_samples: list[float] = []
        self.commit_step_samples: list[int] = []
        self.sync_truncations = 0  # combos whose steps _synchronize_steps reduced
        self.sync_combos = 0  # combos _synchronize_steps looked at
        self.block_cap_hits = 0  # combos capped by _get_step_granularity instead

    # ---- bookkeeping ------------------------------------------------------

    def on_request_submitted(self, plan: RequestExecutionPlan):
        # TetriServe keys step granularity on the IMAGE size (height*width),
        # not on the attention sequence length. Capture it before super()
        # dispatches, since planning may run inside on_request_submitted.
        value = plan.initial_artifacts[0].value if plan.initial_artifacts else None
        height = int(getattr(value, "height", 0) or 0)
        width = int(getattr(value, "width", 0) or 0)
        self._pixels[plan.request_id] = max(1, height * width)
        return super().on_request_submitted(plan)

    def _release_dispatched_task(self, event: WorkerEvent):
        task = self._task_index.get(event.task_id)
        if task is not None and task.kind == TaskKind.DIT_STEP_CHUNK:
            done = 1
            if task.step_range is not None:
                done = max(1, int(task.step_range.end) - int(task.step_range.start))
            left = self._commit_steps.get(event.request_id, 0)
            if left > 0:
                left -= done
                if left <= 0:
                    self._clear_commitment(event.request_id)
                else:
                    self._commit_steps[event.request_id] = left
        super()._release_dispatched_task(event)

    def _drop_request(self, request_id: str) -> None:
        self._clear_commitment(request_id)
        self._pixels.pop(request_id, None)
        super()._drop_request(request_id)

    def _clear_commitment(self, request_id: str) -> None:
        self._commit_steps.pop(request_id, None)
        self._commit_ranks.pop(request_id, None)

    def on_round_tick(self, now_ms: float):
        # The port needs no wall-clock ticks (its round length is internal), but
        # honouring them keeps --round-ms usable as an extra wake-up. A tick is
        # just a re-plan, identical to what an arrival/completion triggers.
        self.now_ms = max(self.now_ms, float(now_ms))
        return self._take_dispatchable_tasks()

    # ---- one planning call == one TetriServe schedule() -------------------

    def _take_dispatchable_tasks(self):
        if self.task_runtime_estimator is None:
            # No cost model -> no deadline math and no perf model; degrade to
            # the base best-fit policy rather than fabricating estimates.
            return EdfBestFitPolicy._take_dispatchable_tasks(self)

        free_ranks = self._free_rank_set()
        out: list[InferenceTask] = []
        ready_dit: list[tuple[str, InferenceTask]] = []
        ready_aux: list[tuple[str, InferenceTask]] = []
        for request_id, queue in self.ready_by_request.items():
            if not queue or self.request_inflight.get(request_id, 0) > 0:
                continue
            task = queue[0]
            if task.group_id is not None:
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

        # Aux stages (text_encode / VAE / finalize). TetriServe has no separate
        # aux tasks -- text encoding runs inside the request's own block on its
        # assigned ranks and is explicitly excluded from the step-scheduling
        # cost model (`is_text_encoding_time_included=False`). Our plan exposes
        # them as separate single-rank tasks, so we dispatch them first in EDF
        # order, identically to TetriServeContinuousPolicy, to keep the two
        # variants comparable. See ALIGNMENT_REPORT.md "residual differences".
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

        # Honour live commitments FIRST: a combo that TetriServe would have
        # dispatched as one multi-step block keeps running on its ranks and is
        # excluded from re-planning (their `running_requests` filter).
        pending: list[tuple[str, InferenceTask]] = []
        for request_id, task in ready_dit:
            if self._commit_steps.get(request_id, 0) <= 0:
                pending.append((request_id, task))
                continue
            ranks = self._commit_ranks.get(request_id, ())
            if ranks and set(ranks).issubset(free_ranks):
                group = self._ensure_group_for_ranks(
                    ranks=tuple(ranks), supported_task_kinds=self._DIT_TASK_KINDS
                )
                self._dispatch(task, request_id, group.group_id, tuple(ranks))
                free_ranks.difference_update(ranks)
                out.append(task)
            else:
                # Its ranks were taken by a same-timestamp dispatch (an artifact
                # of our per-event policy calls, which theirs does not have).
                # Drop the commitment and let the planner re-decide.
                self._clear_commitment(request_id)
                pending.append((request_id, task))

        out.extend(self._plan_round(pending, free_ranks))
        return out

    def _plan_round(
        self, pending: list[tuple[str, InferenceTask]], free_ranks: set[int]
    ) -> list[InferenceTask]:
        """dyn_slo_schedule_global(), one iteration of greedy_bin_packing_exhaustive."""
        if not pending or not free_ranks:
            return []
        world = len(free_ranks)

        candidates = self._candidates(pending, world)
        if not candidates:
            return []

        round_duration = self._round_duration_ms(candidates)
        self.round_duration_samples.append(round_duration)
        if round_duration <= 0.0:
            return []

        combos = self._dp_combo_search(candidates, round_duration, world)
        if not combos:
            return []

        # dyn_slo_schedule_global:1512 -- largest rank group first.
        combos.sort(key=lambda c: c.gpus, reverse=True)
        self._scale_up_to_fill_bubble(combos, candidates, world)
        rank_groups = self._assign_ranks(combos, candidates, free_ranks)
        self._synchronize_steps(combos, candidates)

        out: list[InferenceTask] = []
        for combo, ranks in zip(combos, rank_groups):
            if combo.steps <= 0 or len(ranks) != combo.gpus:
                continue
            cand = candidates[combo.idx]
            # _build_schedule_DAG chops a combo into blocks of
            # _get_step_granularity(seq_len) steps, chained on the shared ranks;
            # the next schedule() replaces the DAG, so only the first block ever
            # runs. Commit exactly that first block.
            block_cap = self._block_step_granularity(cand.pixels)
            steps = min(combo.steps, block_cap)
            if block_cap < combo.steps:
                self.block_cap_hits += 1
            ranks_t = tuple(sorted(ranks))
            group = self._ensure_group_for_ranks(
                ranks=ranks_t, supported_task_kinds=self._DIT_TASK_KINDS
            )
            self._dispatch(cand.task, cand.request_id, group.group_id, ranks_t)
            self._commit_steps[cand.request_id] = int(steps)
            self._commit_ranks[cand.request_id] = ranks_t
            self.commit_step_samples.append(int(steps))
            out.append(cand.task)
        return out

    # ---- stage 1: _filter_and_prepare_requests + _compute_optimal_shapes --

    def _candidates(
        self, pending: list[tuple[str, InferenceTask]], world: int
    ) -> list[_TetriCandidate]:
        """``_filter_and_prepare_requests`` (dyn_step_scheduler.py:860).

        Their ``request_tracker.get_merged_requests()`` walks ``req_id`` in
        insertion (= arrival) order, so the candidate list is FCFS, and it is
        then truncated to the number of FREE GPUs. Note their
        ``_group_requests_by_sequence_length`` result (which would sort by
        latency threshold) is computed at dyn_slo_schedule_global:1409 and never
        read -- it is dead code, so admission order really is FCFS, not EDF.
        """
        if self.admission_order == "edf":
            ordered = sorted(pending, key=lambda rt: self._sort_key(rt[0]))
        else:
            ordered = sorted(pending, key=lambda rt: self._arrival_key(rt[0]))
        if self.candidate_limit == "free":
            ordered = ordered[:world]
        out: list[_TetriCandidate] = []
        for request_id, task in ordered:
            meta = self.request_meta.get(request_id)
            if meta is None:
                continue
            remaining = self._remaining_steps(request_id)
            if remaining <= 0:
                continue
            shape = self._optimal_shape(
                meta.latent_seq_len, remaining, world, self._budget_ms(meta)
            )
            if sum(shape) <= 0:
                continue
            out.append(
                _TetriCandidate(
                    request_id=request_id,
                    task=task,
                    meta=meta,
                    pixels=self._pixels.get(request_id, meta.latent_seq_len),
                    shape=shape,
                    remaining=remaining,
                )
            )
        return out

    def _arrival_key(self, request_id: str):
        meta = self.request_meta.get(request_id)
        if meta is None:
            return (math.inf, 0, request_id)
        return (meta.arrival_ms, meta.order, request_id)

    def _budget_ms(self, meta) -> float:
        """``latency_threshold + arrival_time - get_current_time()``
        (get_optimal_shape:465). Compared against DENOISING time only -- their
        optimal-shape search excludes text encoding, so it is measured against
        the wall-clock deadline without reserving time for the other stages."""
        deadline = meta.deadline_ms
        if deadline is None or not math.isfinite(deadline):
            return math.inf
        return float(deadline) - self.now_ms

    # ---- stage 2: get_optimal_shape_python -------------------------------

    def _optimal_shape(
        self, latent_seq_len: int, num_steps: int, world: int, budget_ms: float
    ) -> list[int]:
        """``get_optimal_shape_python`` (dyn_step_scheduler.py:484).

        Minimize GPU-hours ``sum(r_k * t_k * k)`` over allocations of
        ``num_steps`` denoising steps to the 1/2/4/8-GPU configs, subject to
        ``sum(r_k * t_k) <= budget``; if nothing fits, take the allocation with
        the smallest total time (their ``failed_shape``).

        Their triple ``for`` loop is O(n^3) per request per planning call, which
        is far too slow here (thousands of planning calls). We precompute, per
        (latent_seq_len, num_steps, world), the Pareto staircase of
        (total_time -> min GPU-hours) over the SAME candidate set and answer any
        budget with a bisect. Same enumeration, same objective, same tie-break
        on total time; only the evaluation order differs. ``MIN_STEPS_PER_CONFIG``
        is 1 on non-H100 (dyn_step_scheduler.py:520-523), i.e. inert, so the
        released two-pass retry is a no-op and is not reproduced.
        """
        key = (int(latent_seq_len), int(num_steps), int(world))
        frontier = self._frontier_cache.get(key)
        if frontier is None:
            frontier = self._build_shape_frontier(latent_seq_len, num_steps, world)
            self._frontier_cache[key] = frontier
        times, shapes, fallback = frontier
        if not times:
            return [0, 0, 0, 0]
        idx = bisect.bisect_right(times, budget_ms) - 1
        if idx < 0:
            return list(fallback)
        return list(shapes[idx])

    def _build_shape_frontier(self, latent_seq_len: int, num_steps: int, world: int):
        per_step = []
        for gpus in _GPU_COUNTS:
            cost = self._t_step(latent_seq_len, gpus) if gpus <= world else -1.0
            per_step.append(cost if cost > 0 else -1.0)
        allowed = [i for i, cost in enumerate(per_step) if cost > 0]
        if not allowed or num_steps <= 0:
            return ([], [], (0, 0, 0, 0))

        cands: list[tuple[float, float, tuple[int, ...]]] = []
        cur = [0, 0, 0, 0]

        def recurse(pos: int, left: int) -> None:
            if pos == len(allowed) - 1:
                cur[allowed[pos]] = left
                total = sum(cur[i] * per_step[i] for i in allowed)
                gpu_hours = sum(cur[i] * per_step[i] * _GPU_COUNTS[i] for i in allowed)
                cands.append((total, gpu_hours, tuple(cur)))
                cur[allowed[pos]] = 0
                return
            for value in range(left + 1):
                cur[allowed[pos]] = value
                recurse(pos + 1, left - value)
            cur[allowed[pos]] = 0

        recurse(0, num_steps)
        cands.sort(key=lambda c: (c[0], c[1]))
        times: list[float] = []
        shapes: list[tuple[int, ...]] = []
        best = math.inf
        for total, gpu_hours, shape in cands:
            if gpu_hours < best - 1e-9:
                best = gpu_hours
                if times and abs(times[-1] - total) < 1e-12:
                    shapes[-1] = shape
                else:
                    times.append(total)
                    shapes.append(shape)
        return (times, shapes, cands[0][2])

    def _t_step(self, latent_seq_len: int, gpus: int) -> float:
        """``_get_time_per_step`` / ``get_split_batch_time`` with batch_size 1.

        With bs=1 ``get_split_batch`` returns (1, gpu_num) (splitable=False), so
        ``get_split_batch_time(bs, seq, steps, g) == per_step(g) * steps``. Our
        workloads are one video per request, so bs is always 1 and every
        batch-splitting path in TetriServe is the identity -- see
        ALIGNMENT_REPORT.md on batching.
        """
        key = (int(latent_seq_len), int(gpus))
        cached = self._tstep_cache.get(key)
        if cached is None:
            cached = self._safe_estimate(
                ParallelSpec(tp=1, sp=int(gpus), cfg=1),
                int(latent_seq_len),
                "dit_step_chunk",
            )
            self._tstep_cache[key] = cached
        return cached

    # ---- stage 3: _calculate_round_duration ------------------------------

    def _round_step_granularity(self, pixels: int) -> int:
        """Inline granularity of ``_calculate_round_duration`` (line 971-980).
        NOTE this differs from ``_get_step_granularity`` in the released code."""
        if self.gpu_is_h100:
            return 4 if pixels >= 1024 * 1024 else 10
        return 2 if pixels >= 1024 * 1024 else 20

    def _block_step_granularity(self, pixels: int) -> int:
        """``_get_step_granularity`` (line 1239), used by ``_build_schedule_DAG``
        to chop a combo into executable blocks. Identical on both GPU branches."""
        if pixels > 2048 * 2048:
            return 2
        if pixels >= 1024 * 1024:
            return 4
        return 10

    def _round_duration_ms(self, candidates: list[_TetriCandidate]) -> float:
        """``_calculate_round_duration`` (line 945): the max time any remaining
        request needs to complete ONE step granularity at a GPU count its
        optimal shape actually uses."""
        best = 0.0
        for cand in candidates:
            granularity = self._round_step_granularity(cand.pixels)
            for gi, gpus in enumerate(_GPU_COUNTS):
                if cand.shape[gi] > 0:
                    best = max(
                        best,
                        self._t_step(cand.meta.latent_seq_len, gpus) * granularity,
                    )
        return best

    # ---- stage 4: _dynamic_programming_combo_search ----------------------

    def _survives(self, meta, finish_ms: float) -> bool:
        """Survival predicate of the DP (lines 2032-2041 / 2110-2117), including
        the SLO_TOLERANCE band."""
        deadline = meta.deadline_ms
        if deadline is None or not math.isfinite(deadline):
            return True
        if finish_ms <= deadline:
            return True
        threshold = float(deadline) - float(meta.arrival_ms)
        if threshold <= 0:
            return False
        return abs(finish_ms - deadline) / threshold <= self.slo_tolerance

    def _shape_time_ms(self, cand: _TetriCandidate, shape) -> float:
        return sum(
            self._t_step(cand.meta.latent_seq_len, gpus) * shape[gi]
            for gi, gpus in enumerate(_GPU_COUNTS)
            if shape[gi] > 0
        )

    def _dp_combo_search(
        self, candidates: list[_TetriCandidate], round_duration: float, capacity: int
    ) -> list[_TetriCombo]:
        """``_dynamic_programming_combo_search`` (dyn_step_scheduler.py:1977).

        Options per request: "skip" (0 GPUs) or "schedule at 2^gi GPUs" for each
        config its optimal shape uses, taking as many steps as fit in
        ``round_duration``. Group knapsack over the free-GPU capacity maximizing
        the lexicographic objective (survivors, scheduled count, GPUs used).

        DIVERGENCE: their plan reconstruction walks a per-request back-pointer
        array that is deep-copied forward and can re-select the same request or
        terminate early (lines 2143-2259). We keep their option construction,
        transition and objective verbatim but reconstruct the plan correctly
        from a per-(request, capacity) choice table. A correct reconstruction can
        only realise a schedule at least as good under their own objective, so
        this favours TetriServe.
        """
        now = self.now_ms
        round_end = now + round_duration
        options: list[list[tuple[int, int, int, int]]] = []  # (gpus, steps, surv, sched)
        for cand in candidates:
            opts: list[tuple[int, int, int, int]] = []
            skip_lb = self._shape_time_ms(cand, cand.shape)
            opts.append((0, 0, 1 if self._survives(cand.meta, round_end + skip_lb) else 0, 0))
            for gi, gpus in enumerate(_GPU_COUNTS):
                if cand.shape[gi] == 0:
                    continue
                if gpus > capacity:
                    break
                per_step = self._t_step(cand.meta.latent_seq_len, gpus)
                if per_step <= 0:
                    continue
                steps = min(cand.shape[gi], int(round_duration // per_step))
                if steps <= 0:
                    continue
                after = list(cand.shape)
                after[gi] -= steps
                finish = now + per_step * steps + self._shape_time_ms(cand, after)
                opts.append(
                    (gpus, steps, 1 if self._survives(cand.meta, finish) else 0, 1)
                )
            options.append(opts)

        neg = (-math.inf, -1, -1)
        table: list[tuple] = [neg] * (capacity + 1)
        table[0] = (0.0, 0, 0)
        choice: list[list[tuple[int, int] | None]] = [
            [None] * (capacity + 1) for _ in options
        ]
        for req_idx, opts in enumerate(options):
            nxt: list[tuple] = [neg] * (capacity + 1)
            for used in range(capacity + 1):
                if table[used][0] == -math.inf:
                    continue
                for opt_idx, (gpus, _steps, surv, sched) in enumerate(opts):
                    total = used + gpus
                    if total > capacity:
                        continue
                    cand_state = (
                        table[used][0] + surv,
                        table[used][1] + sched,
                        table[used][2] + gpus,
                    )
                    if cand_state > nxt[total]:
                        nxt[total] = cand_state
                        choice[req_idx][total] = (used, opt_idx)
            table = nxt

        best_capacity = 0
        best_state = table[0]
        for used in range(1, capacity + 1):
            if table[used] > best_state:
                best_state = table[used]
                best_capacity = used

        combos: list[_TetriCombo] = []
        used = best_capacity
        for req_idx in range(len(options) - 1, -1, -1):
            picked = choice[req_idx][used]
            if picked is None:
                break
            prev_used, opt_idx = picked
            gpus, steps, _surv, sched = options[req_idx][opt_idx]
            if sched:
                combos.append(
                    _TetriCombo(
                        idx=req_idx,
                        gpus=gpus,
                        gi=int(math.log2(gpus)),
                        steps=steps,
                    )
                )
            used = prev_used
        combos.reverse()
        return combos

    # ---- stage 5: _scale_up_ranks_to_fill_bubble -------------------------

    def _scale_up_to_fill_bubble(
        self, combos: list[_TetriCombo], candidates: list[_TetriCandidate], world: int
    ) -> None:
        """``_scale_up_ranks_to_fill_bubble`` (dyn_step_scheduler.py:995).

        Enabled unconditionally on non-H100. Doubles a combo's GPU count while
        that is strictly faster per step and still fits, never reaching 8.
        ``rank_used`` accumulates across combos exactly as their
        ``updated_best_rank_used`` does.
        """
        if self.gpu_is_h100:
            return
        rank_used = sum(combo.gpus for combo in combos)
        if rank_used >= world:
            return
        for combo in combos:
            cand = candidates[combo.idx]
            original = combo.gpus
            needed = original
            while rank_used + needed * 2 - original <= world:
                current = self._t_step(cand.meta.latent_seq_len, needed)
                doubled = self._t_step(cand.meta.latent_seq_len, needed * 2)
                if doubled <= 0 or current <= doubled:
                    break
                if needed * 2 == 8:
                    break
                needed *= 2
            if needed != original:
                rank_used += needed - original
                cand.shape[combo.gi] -= combo.steps
                combo.gpus = needed
                combo.gi = int(math.log2(needed))
                cand.shape[combo.gi] += combo.steps

    # ---- stage 6: _assign_ranks_to_combos --------------------------------

    def _assign_ranks(
        self,
        combos: list[_TetriCombo],
        candidates: list[_TetriCandidate],
        free_ranks: set[int],
    ) -> list[list[int]]:
        """``_assign_ranks_to_combos`` (dyn_step_scheduler.py:1076).

        Phase 1 reserves ranks the request already holds, phase 2 fills from the
        free pool in ascending rank order.

        DIVERGENCE: in the released code phase 1 is dead -- ``rank_assign`` is
        re-initialised to all -1 inside ``greedy_bin_packing_exhaustive`` and the
        loop breaks after one iteration, so no rank is ever marked as belonging
        to a request. We implement the intent (placement preservation, which is
        also what our other policies do) because otherwise every combo migrates
        ranks every round and pays our simulator's reshard cost, which their
        model does not have. Reshard here is 3 ms against ~350-5400 ms steps, so
        the effect is negligible either way; set placement_preserving=False
        (TETRISERVE_PLACEMENT=0) to reproduce the released behaviour.
        """
        ordered_free = sorted(free_ranks)
        groups: list[list[int]] = [[] for _ in combos]
        reserved: set[int] = set()
        if self.placement_preserving:
            for pos, combo in enumerate(combos):
                held = self._request_ranks(candidates[combo.idx].request_id)
                if not held:
                    continue
                need = combo.gpus
                for rank in ordered_free:
                    if need == 0:
                        break
                    if rank in held and rank not in reserved:
                        groups[pos].append(rank)
                        reserved.add(rank)
                        need -= 1
        for pos, combo in enumerate(combos):
            need = combo.gpus - len(groups[pos])
            for rank in ordered_free:
                if need == 0:
                    break
                if rank not in reserved:
                    groups[pos].append(rank)
                    reserved.add(rank)
                    need -= 1
        return groups

    def _request_ranks(self, request_id: str) -> set[int]:
        group_id = self.request_last_group.get(request_id)
        if group_id is None:
            return set()
        try:
            return {int(rank) for rank in self.topology.get_group(group_id).ranks}
        except (KeyError, AttributeError):
            return set()

    # ---- stage 7: _adjust_steps_for_synchronization ----------------------

    def _synchronize_steps(
        self, combos: list[_TetriCombo], candidates: list[_TetriCandidate]
    ) -> None:
        """``_adjust_steps_for_synchronization`` (dyn_step_scheduler.py:1135).

        ``rank_time`` is all-zero (it is re-initialised per
        ``greedy_bin_packing_exhaustive`` call, which breaks after one
        iteration), so ``combo_previous_time`` is 0 for every combo and
        ``min_time_required`` collapses to the SHORTEST admitted combo's
        duration. Every other combo is then truncated to the largest step count
        that still fits inside it -- this is precisely the mechanism that
        removes end-of-round idle, and the piece the previous fixed-round
        implementation had no equivalent of. ``max_step`` starts at 1 in the
        released code, so a combo whose single step overruns the window still
        runs one step.
        """
        if not combos:
            return
        min_time = math.inf
        for combo in combos:
            cand = candidates[combo.idx]
            min_time = min(
                min_time,
                self._t_step(cand.meta.latent_seq_len, combo.gpus) * combo.steps,
            )
        for combo in combos:
            cand = candidates[combo.idx]
            per_step = self._t_step(cand.meta.latent_seq_len, combo.gpus)
            max_step = 1
            if per_step > 0:
                for step in range(1, combo.steps + 1):
                    if per_step * step <= min_time + 1e-9:
                        max_step = step
                    else:
                        break
            self.sync_combos += 1
            if max_step < combo.steps:
                self.sync_truncations += 1
            combo.steps = max_step


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
    """Aligned port of the released TetriServe SLO scheduler.

    Env knobs (all default to the faithful setting):
      TETRISERVE_GPU_H100=1        use the H100_HBM3 granularity branch
      TETRISERVE_SLO_TOLERANCE=x   survival-band tolerance (released: 0.07)
      TETRISERVE_PLACEMENT=0       reproduce the released (dead) phase-1 rank
                                   reservation, i.e. no placement preservation
      TETRISERVE_ADMISSION=edf     sensitivity only: EDF instead of the released
                                   FCFS candidate order
      TETRISERVE_CANDIDATES=all    sensitivity only: do not truncate the
                                   candidate list to the free-GPU count
    """
    raw = os.getenv("EDF_GREEDY_SP_SIZES", "").strip()
    allowed_sp_sizes: tuple[int, ...] | None = None
    if raw:
        allowed_sp_sizes = tuple(int(s.strip()) for s in raw.split(",") if s.strip())
    return TetriServeRoundPolicy(
        topology=topology,
        task_runtime_estimator=task_runtime_estimator,
        allowed_sp_sizes=allowed_sp_sizes,
        gpu_is_h100=os.getenv("TETRISERVE_GPU_H100", "0") == "1",
        slo_tolerance=float(os.getenv("TETRISERVE_SLO_TOLERANCE", SLO_TOLERANCE)),
        placement_preserving=os.getenv("TETRISERVE_PLACEMENT", "1") != "0",
        admission_order=os.getenv("TETRISERVE_ADMISSION", "fcfs"),
        candidate_limit=os.getenv("TETRISERVE_CANDIDATES", "free"),
    )


def make_tetriserve_round_fixed(topology, task_runtime_estimator):
    """The OLD, unfaithful fixed-wall-clock-round variant. Requires --round-ms."""
    raw = os.getenv("EDF_GREEDY_SP_SIZES", "").strip()
    allowed_sp_sizes: tuple[int, ...] | None = None
    if raw:
        allowed_sp_sizes = tuple(int(s.strip()) for s in raw.split(",") if s.strip())
    return TetriServeFixedRoundPolicy(
        topology=topology,
        task_runtime_estimator=task_runtime_estimator,
        allowed_sp_sizes=allowed_sp_sizes,
        elastic_scale_up=os.getenv("TETRISERVE_ELASTIC_SCALE_UP", "1") != "0",
    )
