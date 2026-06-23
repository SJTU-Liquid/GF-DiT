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


from benchmarks.diffusion.policies.step_preempt import StepPreemptiveSingleGroupPolicy

class CleanEdfPreemptivePolicy(StepPreemptiveSingleGroupPolicy):
    """Step-boundary preemptive policy scored purely by deadline slack.

    Replaces the scoring stack inherited from the parent class with a single
    rule: for each (request, candidate-group) pair, compute the projected
    finish time if dispatched on that group, and the slack to the deadline.
    Smaller slack -> higher score. That is all. Priority falls out of
    deadlines being tight (S has alpha=1.5 * T_single; L has alpha=6.0).

    No class_base / continue_bonus / demote_bonus / aging_cap. Reshard cost
    enters only through the projected finish time (it pushes the eta).

    Preemption: keeps the same step-boundary mechanism as the parent. The
    rank-DP picks at most one action per request, non-overlapping. When S
    arrives its slack is small (or already negative), so the DP picks S
    over whatever was running.
    """

    def _score_action(
        self,
        *,
        request_id,
        task,
        group_id,
        action,
        task_ms,
        remaining_ms,
        reshard_cost_ms,
    ):
        """Score = (1 + saved) / (1 + miss), all in T_single units.

        Multiplicative form so the score stays strictly positive even when
        a request will miss its deadline by a lot. This matters because
        the DP "skips" any request whose best score is <= 0 (skip costs
        nothing), so a strictly-additive penalty starves S whenever S's
        only feasible action would be late. Multiplicative penalty still
        reorders correctly (smaller miss -> higher score, more saved ->
        higher score) but always dispatches over skipping.

        Components (both dimensionless, in T_single units):

          saved = 1 - remaining_ms / T_single
              Throughput reward: 0 on SP1, grows toward 1.0 on SP8. Drives
              the DP to use larger SP groups when ranks are free. With the
              convex synthetic cost model SP_k is strictly faster for
              every k, so this term naturally encodes "if parallelism is
              not a regression, use more ranks".

          miss  = max(0, eta - deadline) / T_single
              SLO penalty in the same units as saved.

        Tie-breaking: the rank-DP iterates candidates_by_request in
        priority order (S first via _request_sort_key), so equally-good
        slots go to the more urgent request first.
        """
        meta = self.request_meta.get(request_id)
        if meta is None:
            return 0.0
        scale = max(1.0, float(meta.profiled_optimal_latency_ms or remaining_ms or 1.0))
        eta_ms = self.now_ms + remaining_ms + reshard_cost_ms
        saved_norm = max(0.0, 1.0 - remaining_ms / scale)
        if meta.deadline_ms is None:
            return 1.0 + saved_norm
        miss_norm = max(0.0, eta_ms - meta.deadline_ms) / scale
        return (1.0 + saved_norm) / (1.0 + miss_norm)


def make_clean_edf_pause(topology, task_runtime_estimator):
    return CleanEdfPreemptivePolicy(
        topology=topology,
        task_runtime_estimator=task_runtime_estimator,
        mode="pause",
        reshard_ms=float(os.getenv("STEP_PREEMPT_RESHARD_MS", "3")),
        top_k=int(os.getenv("STEP_PREEMPT_TOP_K", "6")),
        demote_max_sp=int(os.getenv("STEP_PREEMPT_DEMOTE_MAX_SP", "2")),
        max_pause_ms=float(os.getenv("STEP_PREEMPT_MAX_PAUSE_MS", "600000")),
        aging_force_dispatch_ms=float(os.getenv("STEP_PREEMPT_AGING_FORCE_DISPATCH_MS", "60000")),
        primal_dual_enable=False,
    )


def make_clean_edf_demote(topology, task_runtime_estimator):
    return CleanEdfPreemptivePolicy(
        topology=topology,
        task_runtime_estimator=task_runtime_estimator,
        mode="demote",
        reshard_ms=float(os.getenv("STEP_PREEMPT_RESHARD_MS", "3")),
        top_k=int(os.getenv("STEP_PREEMPT_TOP_K", "6")),
        demote_max_sp=int(os.getenv("STEP_PREEMPT_DEMOTE_MAX_SP", "2")),
        max_pause_ms=float(os.getenv("STEP_PREEMPT_MAX_PAUSE_MS", "600000")),
        aging_force_dispatch_ms=float(os.getenv("STEP_PREEMPT_AGING_FORCE_DISPATCH_MS", "60000")),
        primal_dual_enable=False,
    )


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------
