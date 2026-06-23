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


from benchmarks.diffusion.policies.edf_greedy import EdfGreedyPolicy

@dataclass(frozen=True)
class _EdfBestFitRequestMeta:
    deadline_ms: float | None
    priority: int
    arrival_ms: float
    order: int
    latent_seq_len: int
    voxels: int
    text_seq_len: int
    num_steps: int
    chunk_steps: int


class EdfBestFitPolicy(EdfGreedyPolicy):
    """EDF + deadline-aware best-fit, then redistribute leftovers.

    Differs from EdfGreedyPolicy (largest-fit) in how it picks the SP
    size for each ready-not-inflight request:

    PASS 1 — best-fit per request in EDF order:
        for each request, pick the smallest allowed_sp_size such that
        the request's estimated end-to-end finish time (now + sum of
        remaining task durations, DiT at sp, aux/VAE at sp1) is <=
        its deadline. If no sp_size meets the deadline, fall back to
        the largest sp_size that fits in the remaining free-rank
        budget (least-bad). If the smallest allowed sp doesn't fit at
        all, the request waits until next tick.

    PASS 2 — redistribute leftovers:
        if the free-rank budget is non-empty after PASS 1, promote
        placed DiT requests (still walked in EDF order) to the next
        allowed_sp_size whose delta fits in the budget. This recovers
        the throughput EdfGreedyPolicy gets from "always largest fit"
        without serializing concurrently-urgent requests.

    PASS 3 — materialize: claim ranks and dispatch with the final
    (sp_size, ranks) per task.

    Falls back to base-class largest-fit when there is no
    task_runtime_estimator (no cost model => no deadline math possible).
    """

    request_meta: dict[str, _EdfBestFitRequestMeta]  # type: ignore[assignment]

    def __init__(
        self,
        topology: RuntimeTopology,
        task_runtime_estimator: TaskRuntimeEstimator | None = None,
        *,
        allowed_sp_sizes: Iterable[int] | None = None,
    ) -> None:
        super().__init__(
            topology=topology,
            task_runtime_estimator=task_runtime_estimator,
            allowed_sp_sizes=allowed_sp_sizes,
        )
        self._request_tasks: dict[str, list[InferenceTask]] = {}

    def on_request_submitted(self, plan: RequestExecutionPlan):
        # Index this request's tasks for fast estimate-finish iteration.
        self._request_tasks[plan.request_id] = list(plan.tasks.values())
        return super().on_request_submitted(plan)

    def _drop_request(self, request_id: str) -> None:
        self._request_tasks.pop(request_id, None)
        super()._drop_request(request_id)

    def _meta_from_value(self, value, order: int) -> _EdfBestFitRequestMeta:
        priority = int(getattr(value, "priority", 0) or 0)
        arrival_ms = float(getattr(value, "arrival_ms", 0.0) or 0.0)
        deadline_raw = getattr(value, "deadline_ms", None)
        deadline_ms = float(deadline_raw) if deadline_raw is not None else None
        self.now_ms = max(self.now_ms, arrival_ms)
        return _EdfBestFitRequestMeta(
            deadline_ms=deadline_ms,
            priority=priority,
            arrival_ms=arrival_ms,
            order=order,
            latent_seq_len=int(getattr(value, "latent_seq_len", 0) or 0),
            voxels=int(getattr(value, "voxels", 0) or 0),
            text_seq_len=int(getattr(value, "text_seq_len", 0) or 0),
            num_steps=int(getattr(value, "num_steps", 0) or 0),
            chunk_steps=int(getattr(value, "denoise_chunk_size", 1) or 1),
        )

    def _take_dispatchable_tasks(self):
        free_ranks = self._free_rank_set()
        ready: list[tuple[str, InferenceTask]] = []
        for request_id, queue in self.ready_by_request.items():
            if not queue:
                continue
            if self.request_inflight.get(request_id, 0) > 0:
                continue
            ready.append((request_id, queue[0]))
        ready.sort(key=lambda rt: self._sort_key(rt[0]))

        out: list[InferenceTask] = []

        # Tasks pinned to a specific group bypass the best-fit machinery:
        # they only fit one way. Materialize them immediately so PASS 1
        # operates on an already-reduced free-rank budget.
        remaining_ready: list[tuple[str, InferenceTask]] = []
        for request_id, task in ready:
            if task.group_id is None:
                remaining_ready.append((request_id, task))
                continue
            placement = self._select_placement(task, free_ranks)
            if placement is None:
                continue
            group_id, ranks = placement
            self._dispatch(task, request_id, group_id, ranks)
            free_ranks.difference_update(ranks)
            out.append(task)

        # PASS 1: pick a tentative sp_size per request (no claims yet).
        plans: list[dict] = []
        budget = len(free_ranks)
        for request_id, task in remaining_ready:
            sp = self._best_fit_sp(request_id, task, budget)
            if sp is None:
                continue
            plans.append({"request_id": request_id, "task": task, "sp_size": sp})
            budget -= sp

        # PASS 2: redistribute leftover budget to most-urgent placed DiT
        # requests. plans preserves EDF order from `remaining_ready`.
        #
        # Only upgrade to a bigger SP when the cost model says it is
        # actually faster for THIS request. On small-latent workloads
        # (Qwen Image S@512: SP1 155 ms, SP4 192 ms / step) collective
        # overhead can make a wider SP slower, so upgrading wastes ranks
        # and slows the request itself. Mirrors the same fix in the
        # production policy at vllm_omni/.../edf_best_fit.py.
        sorted_allowed_asc = sorted(self.allowed_sp_sizes)
        while budget > 0:
            grew = False
            for entry in plans:
                if entry["task"].kind not in self._DIT_TASK_KINDS:
                    continue
                cur = entry["sp_size"]
                request_id = entry["request_id"]
                cur_finish = self._estimate_finish_ms(request_id, cur)
                chosen = None
                for s in sorted_allowed_asc:
                    if s <= cur:
                        continue
                    delta = s - cur
                    if delta > budget:
                        break
                    if self._estimate_finish_ms(request_id, s) < cur_finish:
                        chosen = s
                        break
                if chosen is None:
                    continue
                entry["sp_size"] = chosen
                budget -= chosen - cur
                grew = True
                break  # restart from most-urgent
            if not grew:
                break

        # PASS 3: materialize. Single-rank tasks pick the smallest free
        # rank for stable assignment; DiT tasks take sorted(free)[:sp].
        for entry in plans:
            sp = entry["sp_size"]
            task = entry["task"]
            if task.kind in self._SINGLE_RANK_TASK_KINDS:
                ranks = (min(free_ranks),)
                group = self._ensure_group_for_ranks(
                    ranks=ranks,
                    supported_task_kinds=self._STANDARD_TASK_KINDS,
                )
            else:
                ranks = self._pick_ranks_for_request(
                    entry["request_id"], sp, free_ranks
                )
                group = self._ensure_group_for_ranks(
                    ranks=ranks,
                    supported_task_kinds=self._DIT_TASK_KINDS,
                )
            free_ranks.difference_update(ranks)
            self._dispatch(task, entry["request_id"], group.group_id, ranks)
            out.append(task)
        return out

    # ---- best-fit machinery ----

    def _best_fit_sp(
        self,
        request_id: str,
        task: InferenceTask,
        budget: int,
    ) -> int | None:
        if budget <= 0:
            return None
        if task.kind in self._SINGLE_RANK_TASK_KINDS:
            # Aux stages: nothing to "fit" — always SP1 (or smallest allowed).
            sp = 1 if 1 in self.allowed_sp_sizes else min(self.allowed_sp_sizes)
            return sp if sp <= budget else None
        if task.kind not in self._DIT_TASK_KINDS:
            return None
        candidates = sorted(s for s in self.allowed_sp_sizes if s <= budget)
        if not candidates:
            return None
        meta = self.request_meta.get(request_id)
        # No deadline / no estimator: fall back to the base policy's
        # largest-fit behavior so EdfBestFitPolicy reduces to EdfGreedy.
        if (
            meta is None
            or meta.deadline_ms is None
            or not math.isfinite(meta.deadline_ms)
            or self.task_runtime_estimator is None
        ):
            return candidates[-1]
        for sp in candidates:
            if self._estimate_finish_ms(request_id, sp) <= meta.deadline_ms:
                return sp
        # No sp meets the deadline (request is doomed): pick the sp that
        # FINISHES SOONEST per the cost model, not blindly the largest or
        # smallest. Cost-aware argmin gives S -> SP1 (narrow fastest for small,
        # avoids wide-SP waste) and L -> SP4 (wide fastest for large, clears the
        # doomed job ASAP). Largest-fit over-widens L to SP8 (slower than SP4);
        # smallest-fit under-widens L to SP1 (much slower).
        return min(candidates, key=lambda s: self._estimate_finish_ms(request_id, s))

    def _estimate_finish_ms(self, request_id: str, sp_size: int) -> float:
        meta = self.request_meta.get(request_id)
        if meta is None or self.task_runtime_estimator is None:
            return math.inf
        sp1 = ParallelSpec(tp=1, sp=1, cfg=1)
        dit_spec = ParallelSpec(tp=1, sp=sp_size, cfg=1)
        total_ms = 0.0
        for t in self._request_tasks.get(request_id, ()):
            if t.status == TaskStatus.FINISHED:
                continue
            kind = t.kind
            if kind == TaskKind.DIT_STEP_CHUNK:
                chunk_steps = 1
                if t.step_range is not None:
                    chunk_steps = max(1, t.step_range.end - t.step_range.start)
                stage = str(t.payload.get("cost_model_stage") or "dit_step_chunk")
                total_ms += self._safe_estimate(
                    dit_spec, meta.latent_seq_len, stage,
                ) * chunk_steps
            elif kind == TaskKind.DIT_PREPARE:
                total_ms += self._safe_estimate(
                    dit_spec, meta.latent_seq_len, "dit_prepare",
                )
            elif kind == TaskKind.TIMESTEP_PREPARE:
                total_ms += self._safe_estimate(
                    dit_spec, meta.num_steps, "timestep_prepare",
                )
            elif kind == TaskKind.TEXT_ENCODE:
                total_ms += self._safe_estimate(sp1, meta.text_seq_len, "text_encode")
            elif kind == TaskKind.VAE_DECODE:
                total_ms += self._safe_estimate(sp1, meta.voxels, "vae_decode")
            elif kind == TaskKind.FINALIZE:
                total_ms += self._safe_estimate(sp1, 1, "finalize")
        return self.now_ms + total_ms

    def _safe_estimate(self, parallel_spec, seq_len, stage: str) -> float:
        try:
            return float(self.task_runtime_estimator(parallel_spec, int(seq_len), stage))
        except Exception:
            return 0.0


def make_edf_best_fit(topology, task_runtime_estimator):
    raw = os.getenv("EDF_GREEDY_SP_SIZES", "").strip()
    allowed_sp_sizes: tuple[int, ...] | None = None
    if raw:
        allowed_sp_sizes = tuple(int(s.strip()) for s in raw.split(",") if s.strip())
    return EdfBestFitPolicy(
        topology=topology,
        task_runtime_estimator=task_runtime_estimator,
        allowed_sp_sizes=allowed_sp_sizes,
    )


# ---------------------------------------------------------------------------
# Clean EDF policy: no class_base, no continue bonus, no per-class buckets.
# ---------------------------------------------------------------------------
