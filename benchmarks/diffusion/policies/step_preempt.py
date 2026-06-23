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


from benchmarks.diffusion.runtime_v2_step_preemptive_policy import (
    StepBoundaryPreemptiveElasticPolicy,
)
from benchmarks.diffusion.policies.common import (
    _request_class_from_value,
    _groups_with_sp,
)

class StepPreemptiveSingleGroupPolicy(StepBoundaryPreemptiveElasticPolicy):
    """Step-boundary preemptive policy that consumes single-group plans.

    The plan still pins every task to one initial group (chosen by
    ``_pick_initial_dit_group``), but the rank-DP scheduler is free to
    reroute DiT step chunks to a different group at boundaries.  This lets
    L start on the latency group (SP4) and be demoted to SP1 once an S
    burst arrives.

    Two anti-starvation mechanisms layered on top of the parent class:

    1. **Aging force-dispatch**: a paused L whose continuous pause time
       exceeds ``aging_force_dispatch_ms`` is force-dispatched on a free
       SP1 lane in a pre-pass *before* the rank-DP runs. This bypasses
       the score-based competition with S (whose class_base of 100k the
       original aging penalty of at most 15k could not match).

    2. **Continuous primal-dual** Lagrangian multipliers (``lambda_c``)
       updated every scheduling tick based on observed predicted-late
       rate, plugged into the DP scoring as
       ``score += lambda_c * predicted_late_factor``. This is the
       primal-dual layer described in the spec; the parent class only
       updates ``virtual_queues`` on REQUEST_FINISHED (way too slow).
    """

    def __init__(
        self,
        topology: RuntimeTopology,
        task_runtime_estimator,
        *,
        mode: str,
        aging_force_dispatch_ms: float = 60_000.0,
        primal_dual_enable: bool = True,
        primal_dual_step_size: float = 0.05,
        primal_dual_target_late_rate: dict[str, float] | None = None,
        primal_dual_score_gain: float = 50_000.0,
        **kwargs,
    ) -> None:
        super().__init__(
            topology=topology,
            task_runtime_estimator=task_runtime_estimator,
            mode=mode,
            **kwargs,
        )
        self._cursor_s = 0
        self._cursor_m = 0
        self._cursor_l = 0
        # Anti-starvation: a paused L is force-dispatched on an SP1 lane
        # once its continuous pause exceeds this threshold.
        self.aging_force_dispatch_ms = float(aging_force_dispatch_ms)
        # Continuous primal-dual layer.
        self.primal_dual_enable = bool(primal_dual_enable)
        self.primal_dual_step_size = float(primal_dual_step_size)
        self.primal_dual_score_gain = float(primal_dual_score_gain)
        # Target ε_c = fraction of class c we are OK with missing SLO. A
        # smaller ε means the dual variable grows faster when late, which
        # means the scheduler prefers that class harder.
        defaults_eps = {"S": 0.01, "M": 0.05, "L": 0.20}
        if primal_dual_target_late_rate:
            defaults_eps.update({k.upper(): float(v) for k, v in primal_dual_target_late_rate.items()})
        self.primal_dual_target_late_rate = defaults_eps
        # Continuous-time Lagrangian multipliers, updated every scheduling tick.
        self.lambda_c: dict[str, float] = {"S": 0.0, "M": 0.0, "L": 0.0}
        self._last_pd_update_ms: float | None = None

    def _pick_initial_dit_group(self, plan: RequestExecutionPlan) -> str:
        klass = _request_class_from_value(plan)
        if klass == "S":
            self._cursor_s += 1
            return self._latency_group_ids[(self._cursor_s - 1) % len(self._latency_group_ids)]
        if klass == "M":
            self._cursor_m += 1
            return self._small_group_ids[(self._cursor_m - 1) % len(self._small_group_ids)]
        # L: start on latency group so the policy can demote it when S arrives.
        self._cursor_l += 1
        return self._latency_group_ids[(self._cursor_l - 1) % len(self._latency_group_ids)]

    def build_sim_plan(self, *, request, default_builder):
        if not callable(default_builder):
            raise TypeError("default_builder must be callable")
        probe = default_builder(request)
        initial_group_id = self._pick_initial_dit_group(probe)
        # Single-group plan: every task pinned to initial_group_id.  The
        # simulator routes text/VAE/finalize to rank[0] of that group
        # internally; the rank-DP scheduler reassigns later DiT chunks.
        return default_builder(request, group_id=initial_group_id)

    def _take_dispatchable_tasks(self) -> list[InferenceTask]:
        self._update_lambda_c()
        forced = self._force_dispatch_starving_l()
        out = list(forced)
        out.extend(super()._take_dispatchable_tasks())
        return out

    # ------------------------------------------------------------------
    # Dynamic free-pool candidate selection
    # ------------------------------------------------------------------
    #
    # The base class enumerates *all* matching groups for each task, which
    # turns C(8,4)=70 SP4 lanes into 70 candidates per request. The DP
    # then wastes its capacity choosing among overlapping rank masks. The
    # right abstraction is: "I want SP=k. Are there k free ranks anywhere?"
    # If yes, pick *one* group whose ranks are free; if no, fall back to
    # smaller SP. This collapses 70 candidates to one per SP degree.

    def _free_rank_set(self) -> set[int]:
        return {
            int(group.ranks[0])
            for group in self.topology.groups
            if int(group.parallel_spec.sp) == 1 and len(group.ranks) == 1
            and self.outstanding_per_rank.get(int(group.ranks[0]), 0) == 0
            and self.reshard_holds_by_rank.get(int(group.ranks[0]), 0) == 0
        }

    def _best_group_for_sp(self, sp: int, free_ranks: set[int], source_group_id: str | None) -> str | None:
        """Pick an SP=sp group whose ranks are subset of ``free_ranks`` ∪ source.

        Source ranks are included because a continue-action on the same
        group does not require those ranks to be free of *itself*. Preference
        order:
          1. The source group, if it has the requested SP and all ranks free.
          2. Any group with all ranks in free_ranks; among those, prefer
             the one whose ``len(ranks ∩ source_ranks)`` is largest so
             reshard cost is smallest.
        """
        source_ranks: set[int] = set()
        if source_group_id is not None:
            try:
                source_ranks = {int(r) for r in self.topology.get_group(source_group_id).ranks}
            except KeyError:
                source_ranks = set()
        usable = free_ranks | source_ranks
        candidates = []
        for group in self.topology.groups:
            if int(group.parallel_spec.sp) != sp:
                continue
            ranks = {int(r) for r in group.ranks}
            if not ranks.issubset(usable):
                continue
            overlap_with_source = len(ranks & source_ranks)
            # Prefer source-group exactly (continue), then most rank-overlap.
            score = (
                0 if group.group_id == source_group_id else 1,
                -overlap_with_source,
                self._group_order.get(group.group_id, 0),
            )
            candidates.append((score, group.group_id))
        if not candidates:
            return None
        candidates.sort()
        return candidates[0][1]

    def _candidate_group_ids(self, request_id: str, task: InferenceTask) -> tuple[str, ...]:
        meta = self.request_meta[request_id]
        source = self.request_last_group.get(request_id)
        pressure = self._foreground_pressure()
        free_ranks = self._free_rank_set()

        ordered: list[str] = []

        def add(group_id: str | None) -> None:
            if group_id is not None and group_id not in ordered:
                ordered.append(group_id)

        # Compute one candidate per SP degree from the free pool. Cap at
        # latency-group SP (max in topology).
        max_sp = max(int(g.parallel_spec.sp) for g in self.topology.groups)
        sp_choices: list[int]
        if meta.request_class == "S":
            # Foreground: prefer highest SP, then fall back to smaller.
            sp_choices = sorted({max_sp, max(2, max_sp // 2), 1}, reverse=True)
        elif meta.request_class == "L" and pressure and task.kind == TaskKind.DIT_STEP_CHUNK:
            # L under foreground pressure: pause or demote behavior.
            if self.mode == "pause" and not self._pause_aged_out(request_id):
                return ()
            if self.mode == "demote":
                sp_choices = [1, max(2, self.demote_max_sp)]
                if self._pause_aged_out(request_id):
                    # Aged out: any SP is allowed so it can promote later.
                    sp_choices = sorted({max_sp, 2, 1}, reverse=True)
            else:
                sp_choices = []
        else:
            # M without pressure / L without pressure / S etc: prefer latency
            # group when free ranks allow it, fall back to small.
            sp_choices = sorted({max_sp, max(2, max_sp // 2), 1}, reverse=True)
        # Always allow continuing on the source group (no reshard).
        if source is not None:
            add(source)
        for sp in sp_choices:
            add(self._best_group_for_sp(sp, free_ranks, source))
        return tuple(ordered)

    # ------------------------------------------------------------------
    # Promotion bonus: kill the bubble where L sits on SP1 while ranks idle
    # ------------------------------------------------------------------

    def _score_action(self, *, request_id, task, group_id, action, task_ms, remaining_ms, reshard_cost_ms):
        base_score = super()._score_action(
            request_id=request_id,
            task=task,
            group_id=group_id,
            action=action,
            task_ms=task_ms,
            remaining_ms=remaining_ms,
            reshard_cost_ms=reshard_cost_ms,
        )
        # Bonus for promoting a request to a higher-SP group when there
        # is no foreground pressure and free latency capacity exists. The
        # parent class already gives +750 for "continue" — that is what
        # was keeping demoted L permanently on SP1 while other ranks
        # sat idle. We add a counter-bonus when promotion makes sense.
        meta = self.request_meta.get(request_id)
        if (
            action == "promote"
            and not self._foreground_pressure()
            and meta is not None
            and meta.request_class != "S"
        ):
            # Bonus large enough to outweigh the +750 continue bonus and
            # the reshard cost, without overpowering deadline/PD terms.
            base_score += 2000.0
        # Primal-dual contribution (same as before)
        if not self.primal_dual_enable:
            return base_score
        if meta is None or meta.deadline_ms is None:
            return base_score
        slack = meta.deadline_ms - self.now_ms - remaining_ms - reshard_cost_ms
        if slack >= 0:
            late_factor = 0.0
        else:
            scale = max(1.0, float(meta.profiled_optimal_latency_ms or remaining_ms or 1.0))
            late_factor = min(2.0, -slack / scale)
        lam = self.lambda_c.get(meta.request_class, 0.0)
        return base_score + self.primal_dual_score_gain * lam * late_factor

    # ------------------------------------------------------------------
    # Primal-dual: continuous-time Lagrangian multipliers
    # ------------------------------------------------------------------

    def _update_lambda_c(self) -> None:
        """Continuous-time update of the dual variable lambda_c per class.

        For each class c at this scheduling tick:

            predicted_late_rate_c = (# of class-c requests currently
                projected to miss their deadline) / max(1, # in-flight or
                pending).

            lambda_c <- max(0, lambda_c + step_size * (predicted_late_rate_c - eps_c))

        ``predicted_late_rate_c`` uses ``_estimate_remaining_ms`` so the
        signal is forward-looking, not just based on already-finished
        requests. This is what makes the dual variable actually drive
        scheduling decisions before the SLO is breached.
        """
        if not self.primal_dual_enable:
            return
        for klass in ("S", "M", "L"):
            late_rate, in_flight = self._estimate_class_late_rate(klass)
            eps = self.primal_dual_target_late_rate.get(klass, 0.05)
            if in_flight == 0:
                # Decay lambda when no work to track for this class.
                self.lambda_c[klass] = max(0.0, self.lambda_c[klass] - self.primal_dual_step_size * eps)
                continue
            self.lambda_c[klass] = max(
                0.0,
                self.lambda_c[klass] + self.primal_dual_step_size * (late_rate - eps),
            )
        self._last_pd_update_ms = self.now_ms

    def _estimate_class_late_rate(self, klass: str) -> tuple[float, int]:
        """Return (predicted_late_fraction, count) for ``klass``."""
        late = 0
        total = 0
        for request_id, meta in self.request_meta.items():
            if meta.request_class != klass:
                continue
            if meta.deadline_ms is None:
                continue
            queue = self.ready_by_request.get(request_id)
            inflight = self.request_inflight.get(request_id, 0)
            if not queue and inflight <= 0:
                continue
            total += 1
            # Project remaining wall time using a generic SP4 estimate to
            # approximate "if I keep running at latency-group SP".
            remaining_ms = self._project_remaining_ms(request_id, queue)
            if self.now_ms + remaining_ms > meta.deadline_ms:
                late += 1
        if total == 0:
            return 0.0, 0
        return late / total, total

    def _project_remaining_ms(self, request_id: str, queue) -> float:
        """Rough estimate of remaining wall time for a request."""
        # Use the request's metadata to count remaining DiT steps.
        if not queue:
            return 0.0
        task = queue[0]
        metadata = dict(task.payload.get("request_metadata", {}))
        num_steps = int(metadata.get("num_steps", 1))
        completed = self.completed_steps.get(request_id, 0)
        remaining_steps = max(0, num_steps - completed)
        if remaining_steps == 0:
            return self._estimate_task_ms(task, task.group_id or self._latency_group_ids[0])
        # Per-step time on latency group as a baseline projection.
        group_id = task.group_id or self._latency_group_ids[0]
        per_step_ms = self._estimate_task_ms(task, group_id)
        # If this is a DIT_STEP_CHUNK with chunk_steps>1, divide.
        if task.kind == TaskKind.DIT_STEP_CHUNK and task.step_range is not None:
            chunk_steps = max(1, task.step_range.end - task.step_range.start)
            per_step_ms /= chunk_steps
        return remaining_steps * per_step_ms

    def _candidate_actions(self, request_id, task):
        """Same as parent but only claims rank[0] for single-rank stages.

        The simulator routes TEXT_ENCODE / VAE_DECODE / FINALIZE to a
        single rank inside the assigned group (see `_task_resource`),
        but the parent class's _candidate_actions claims *all* of the
        group's ranks. That over-claim made the rank-DP refuse to
        co-dispatch other requests whose own text_encode would have
        landed on a disjoint rank-0 of the same SP4 lane, even though
        the simulator would have happily run them in parallel.
        """
        from benchmarks.diffusion.runtime_v2_step_preemptive_policy import (
            ActionCandidate,
            _rank_mask,
        )
        out: list[ActionCandidate] = []
        source_group_id = self.request_last_group.get(request_id)
        single_rank_kind = task.kind in (
            TaskKind.TEXT_ENCODE,
            TaskKind.VAE_DECODE,
            TaskKind.FINALIZE,
        )
        for group_id in self._candidate_group_ids(request_id, task):
            group = self.topology.get_group(group_id)
            if task.kind not in group.supported_task_kinds:
                continue
            target_ranks = (
                (int(group.ranks[0]),) if single_rank_kind else tuple(int(r) for r in group.ranks)
            )
            claimed_ranks = target_ranks
            reshard_cost_ms = 0.0
            if (
                source_group_id is not None
                and source_group_id != group_id
                and not single_rank_kind
            ):
                src_ranks = tuple(int(r) for r in self.topology.get_group(source_group_id).ranks)
                claimed_ranks = tuple(sorted(set(target_ranks) | set(src_ranks)))
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

    def _force_dispatch_starving_l(self) -> list[InferenceTask]:
        """Dispatch any L whose continuous pause exceeds the aging threshold.

        This bypasses ``_should_pause`` and ``_solve_rank_dp``: we directly
        place L on a free SP1 lane to guarantee at least one step of
        progress per ``aging_force_dispatch_ms``. Concurrent foreground
        keeps running on its existing SP4 lane (different ranks).
        """
        if self.aging_force_dispatch_ms <= 0.0:
            return []
        out: list[InferenceTask] = []
        # Build set of free SP1 lane ranks (current scheduling tick).
        sp1_lanes: list[tuple[str, int]] = []
        for group in self.topology.groups:
            if int(group.parallel_spec.sp) != 1 or len(group.ranks) != 1:
                continue
            rank = int(group.ranks[0])
            if self.outstanding_per_rank.get(rank, 0) > 0:
                continue
            if self.reshard_holds_by_rank.get(rank, 0) > 0:
                continue
            sp1_lanes.append((group.group_id, rank))
        if not sp1_lanes:
            return []
        # Iterate L's sorted by pause-since (longest first).
        for request_id in sorted(
            (rid for rid, meta in self.request_meta.items()
             if meta.request_class == "L" and meta.preemptible),
            key=lambda rid: self.paused_since_ms.get(rid, math.inf),
        ):
            if not sp1_lanes:
                break
            paused_at = self.paused_since_ms.get(request_id)
            if paused_at is None:
                continue
            if self.now_ms - paused_at < self.aging_force_dispatch_ms:
                continue
            if self.request_inflight.get(request_id, 0) > 0:
                continue
            queue = self.ready_by_request.get(request_id)
            if not queue:
                continue
            task = queue[0]
            if task.kind != TaskKind.DIT_STEP_CHUNK:
                continue
            target_group_id, target_rank = sp1_lanes.pop(0)
            # Pop the task and dispatch via the existing helpers so all
            # bookkeeping stays consistent with the parent class.
            queue.popleft()
            if not queue:
                self.ready_by_request.pop(request_id, None)
            self._resume(request_id)
            task.group_id = target_group_id
            task.status = TaskStatus.DISPATCHED
            ranks = (target_rank,)
            self._claim_ranks(ranks, reshard=False)
            self.outstanding_per_group[target_group_id] = self.outstanding_per_group.get(target_group_id, 0) + 1
            self.dispatched_kind[task.task_id] = target_group_id
            self.dispatched_ranks_by_task[task.task_id] = ranks
            self._claim_ranks_by_task[task.task_id] = ranks
            self.request_inflight[request_id] = self.request_inflight.get(request_id, 0) + 1
            out.append(task)
        return out


def make_step_preempt_pause(topology, task_runtime_estimator):
    return StepPreemptiveSingleGroupPolicy(
        topology=topology,
        task_runtime_estimator=task_runtime_estimator,
        mode="pause",
        reshard_ms=float(os.getenv("STEP_PREEMPT_RESHARD_MS", "3")),
        top_k=int(os.getenv("STEP_PREEMPT_TOP_K", "6")),
        demote_max_sp=int(os.getenv("STEP_PREEMPT_DEMOTE_MAX_SP", "2")),
        max_pause_ms=float(os.getenv("STEP_PREEMPT_MAX_PAUSE_MS", "600000")),
        aging_force_dispatch_ms=float(os.getenv("STEP_PREEMPT_AGING_FORCE_DISPATCH_MS", "60000")),
    )


def make_step_preempt_demote(topology, task_runtime_estimator):
    return StepPreemptiveSingleGroupPolicy(
        topology=topology,
        task_runtime_estimator=task_runtime_estimator,
        mode="demote",
        reshard_ms=float(os.getenv("STEP_PREEMPT_RESHARD_MS", "3")),
        top_k=int(os.getenv("STEP_PREEMPT_TOP_K", "6")),
        demote_max_sp=int(os.getenv("STEP_PREEMPT_DEMOTE_MAX_SP", "2")),
        max_pause_ms=float(os.getenv("STEP_PREEMPT_MAX_PAUSE_MS", "600000")),
        aging_force_dispatch_ms=float(os.getenv("STEP_PREEMPT_AGING_FORCE_DISPATCH_MS", "60000")),
    )
