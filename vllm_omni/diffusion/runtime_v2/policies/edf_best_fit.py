# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from vllm.logger import init_logger

from vllm_omni.diffusion.runtime_v2._env import env_flag
from vllm_omni.diffusion.runtime_v2.policies.common import TaskRuntimeEstimator
from vllm_omni.diffusion.runtime_v2.policies.edf_greedy import (
    EdfGreedySchedulerPolicy,
    _EdfGreedyRequestMeta,
)
from vllm_omni.diffusion.runtime_v2.protocol import (
    InferenceTask,
    ParallelSpec,
    RequestExecutionPlan,
    TaskKind,
    TaskStatus,
)
from vllm_omni.diffusion.runtime_v2.topology import RuntimeTopology

logger = init_logger(__name__)


@dataclass(frozen=True)
class _EdfBestFitRequestMeta(_EdfGreedyRequestMeta):
    """Adds request-shape fields needed to estimate end-to-end finish time."""

    latent_seq_len: int = 0
    voxels: int = 0
    text_seq_len: int = 0
    num_steps: int = 0
    chunk_steps: int = 1


class EdfBestFitSchedulerPolicy(EdfGreedySchedulerPolicy):
    """EDF + deadline-aware best-fit with surplus redistribution.

    Production port of the simulator's EdfBestFitPolicy
    (``benchmarks/diffusion/policies/edf_best_fit.py``). Each dispatch tick
    runs three passes:

      1. PASS 1 — best-fit per ready-not-inflight request in EDF order:
         for DiT tasks, pick the smallest ``allowed_sp_size`` such that the
         estimated end-to-end finish time (now + sum of remaining task
         durations) is <= the request's deadline. If no sp meets the
         deadline, fall back to the largest sp that fits in the remaining
         free-rank budget. Single-rank stages (TEXT_ENCODE / VAE_DECODE /
         FINALIZE) always take sp=1.

      2. PASS 2 — surplus redistribution: while the free-rank budget is
         non-empty, promote placed DiT requests (still in EDF order) to
         the next allowed_sp_size whose delta fits. This recovers the
         throughput EdfGreedy gets from "always largest fit" without
         serializing concurrently-urgent requests.

      3. PASS 3 — materialize: claim ranks and dispatch.

    Required inputs for best-fit math:
      * ``task_runtime_estimator``: a cost-model-backed callable (see
        ``vllm_omni/diffusion/runtime_v2/cost_model.py``). Without one the
        policy degrades to the parent's largest-fit behavior.
      * Per-request shape metadata in ``plan.metadata`` (``latent_seq_len``,
        ``voxels``, ``num_steps``, ``denoise_chunk_size``, ``text_seq_len``)
        or in ``sampling_params.extra_args``. Missing fields default to 0
        and degrade gracefully (estimated time becomes 0 for that stage).
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
        # (sp, stage) pairs already warned about a failed cost estimate, so the
        # warning in _safe_estimate fires once instead of every dispatch tick.
        self._estimate_warned: set[tuple[Any, str]] = set()
        # Env-gated per-tick dispatch tracing: logs, for every dispatch tick,
        # the free ranks, the EDF-ordered ready set (with per-request latent
        # size + deadline so S/M/L are identifiable), and which requests got
        # dispatched vs passed over. Used to root-cause why a tight-deadline
        # request waits. Off by default (hot path).
        self._dispatch_trace = env_flag("VLLM_RUNTIME_V2_DISPATCH_TRACE", False)

    # ---- lifecycle ----

    def on_request_submitted(self, plan: RequestExecutionPlan):
        # Index this request's tasks for fast finish-time iteration.
        self._request_tasks[plan.request_id] = list(plan.tasks.values())
        return super().on_request_submitted(plan)

    def _drop_request(self, request_id: str) -> None:
        self._request_tasks.pop(request_id, None)
        super()._drop_request(request_id)

    def _meta_from_plan(self, plan: RequestExecutionPlan, order: int) -> _EdfBestFitRequestMeta:
        base = super()._meta_from_plan(plan, order)
        value = plan.initial_artifacts[0].value if plan.initial_artifacts else None
        meta_src = plan.metadata or {}

        def lookup_int(*names: str, default: int = 0) -> int:
            for name in names:
                if name in meta_src and meta_src[name] is not None:
                    try:
                        return int(meta_src[name])
                    except (TypeError, ValueError):
                        continue
            raw = self._metadata_number(value, names, default=float("nan"))
            if not math.isnan(raw):
                try:
                    return int(raw)
                except (TypeError, ValueError):
                    pass
            return int(default)

        return _EdfBestFitRequestMeta(
            deadline_ms=base.deadline_ms,
            priority=base.priority,
            arrival_ms=base.arrival_ms,
            order=base.order,
            latent_seq_len=lookup_int("latent_seq_len"),
            voxels=lookup_int("voxels"),
            text_seq_len=lookup_int("text_seq_len", default=512),
            num_steps=lookup_int("num_steps"),
            chunk_steps=lookup_int("denoise_chunk_size", "chunk_size", default=1),
        )

    # ---- scheduling core ----

    def _take_dispatchable_tasks(self):
        free_ranks = self._free_rank_set()
        trace_free0 = sorted(free_ranks) if self._dispatch_trace else None
        trace_inflight: list[str] = []
        ready: list[tuple[str, InferenceTask]] = []
        for request_id, queue in self.ready_by_request.items():
            if not queue:
                continue
            if self.request_inflight.get(request_id, 0) > 0:
                if self._dispatch_trace:
                    trace_inflight.append(request_id)
                continue
            ready.append((request_id, queue[0]))
        ready.sort(key=lambda rt: self._request_sort_key(rt[0]))

        out: list[InferenceTask] = []

        # Tasks pinned to an explicit group bypass best-fit; materialize them
        # eagerly so PASS 1 operates on the reduced free-rank budget.
        remaining_ready: list[tuple[str, InferenceTask]] = []
        for request_id, task in ready:
            if task.group_id is None:
                remaining_ready.append((request_id, task))
                continue
            placement = self._select_placement(task, free_ranks)
            if placement is None:
                continue
            group_id, ranks = placement
            self._dispatch_task(task, request_id, group_id, ranks)
            free_ranks.difference_update(ranks)
            out.append(task)

        # PASS 1: tentative sp_size per remaining request, no claims yet.
        plans: list[dict[str, Any]] = []
        budget = len(free_ranks)
        for request_id, task in remaining_ready:
            sp = self._best_fit_sp(request_id, task, budget)
            if sp is None:
                continue
            plans.append({"request_id": request_id, "task": task, "sp_size": sp})
            budget -= sp

        # PASS 2: redistribute remaining budget to most-urgent DiT placement
        # (plans preserves EDF order from remaining_ready).
        #
        # Only upgrade to a bigger SP if the cost model says it's actually
        # faster than the current SP for THIS request. For Qwen Image and
        # other small-latent workloads, wider SP can be SLOWER per step
        # because the GFC/NCCL collective overhead dominates the per-rank
        # compute savings (e.g., Qwen S@512x512: SP1 155ms, SP2 161ms, SP4
        # 192ms per step). Upgrading those to a wider SP both costs more
        # ranks AND makes the request itself slower -- pure loss.
        sorted_allowed_asc = sorted(self.allowed_sp_sizes)
        while budget > 0:
            grew = False
            for entry in plans:
                if entry["task"].kind not in self._DIT_TASK_KINDS:
                    continue
                cur = entry["sp_size"]
                request_id = entry["request_id"]
                cur_finish = self._estimate_finish_ms(request_id, cur)
                # Smallest bigger sp that BOTH fits the budget and is actually
                # faster than the current sp for this request.
                chosen = None
                for s in sorted_allowed_asc:
                    if s <= cur:
                        continue
                    delta = s - cur
                    if delta > budget:
                        break  # candidates are sorted; later are even larger
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

        # PASS 3: materialize.
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
            self._dispatch_task(task, entry["request_id"], group.group_id, ranks)
            out.append(task)
        if self._dispatch_trace:
            self._emit_dispatch_trace(trace_free0, ready, out, trace_inflight)
        return out

    def _fmt_req(self, request_id: str, task: InferenceTask | None = None) -> str:
        meta = self.request_meta.get(request_id)
        lat = getattr(meta, "latent_seq_len", 0) if meta is not None else 0
        deadline = getattr(meta, "deadline_ms", math.inf) if meta is not None else math.inf
        label = f"{request_id[-6:]}/lat{lat}/dl{deadline:.0f}"
        if task is not None:
            step = f"#{task.step_range.start}" if task.step_range is not None else ""
            label += f"/{task.kind.value}{step}"
        return label

    def _emit_dispatch_trace(
        self,
        free0: list[int] | None,
        ready: list[tuple[str, "InferenceTask"]],
        out: list["InferenceTask"],
        inflight_skipped: list[str],
    ) -> None:
        """One line per dispatch tick. ready_edf is in EDF order; each entry is
        prefixed '+' if dispatched this tick, '-' if passed over. Cross-check
        free_ranks to see whether a passed-over request lost on ranks or order."""
        if not ready and not out:
            return
        dispatched_ids = {task.task_id for task in out}
        ready_parts = [
            f"{'+' if task.task_id in dispatched_ids else '-'}{self._fmt_req(request_id, task)}"
            for request_id, task in ready
        ]
        logger.info(
            "runtime_v2 dispatch trace: now_ms=%.0f free_ranks=%s n_ready=%d "
            "inflight_skipped=%d ready_edf=[%s]",
            self.now_ms,
            free0,
            len(ready),
            len(inflight_skipped),
            " ".join(ready_parts),
        )

    # ---- best-fit math ----

    def _best_fit_sp(
        self,
        request_id: str,
        task: InferenceTask,
        budget: int,
    ) -> int | None:
        if budget <= 0:
            return None
        if task.kind in self._SINGLE_RANK_TASK_KINDS:
            sp = 1 if 1 in self.allowed_sp_sizes else min(self.allowed_sp_sizes)
            return sp if sp <= budget else None
        if task.kind not in self._DIT_TASK_KINDS:
            return None
        candidates = sorted(s for s in self.allowed_sp_sizes if s <= budget)
        if not candidates:
            return None
        meta = self.request_meta.get(request_id)
        if (
            meta is None
            or not math.isfinite(meta.deadline_ms)
            or self.task_runtime_estimator is None
        ):
            # Degrade to parent's largest-fit.
            return candidates[-1]
        for sp in candidates:
            if self._estimate_finish_ms(request_id, sp) <= meta.deadline_ms:
                return sp
        # No sp meets the deadline (request is doomed): pick the sp that
        # FINISHES SOONEST per the cost model, not blindly the largest or
        # smallest. Cost-aware argmin gives S -> SP1 (narrow is fastest for
        # small, avoids wide-SP waste) and L -> SP4 (wide is fastest for large,
        # clears the doomed job from the system ASAP). Largest-fit over-widens L
        # to SP8 (slower than SP4); smallest-fit under-widens L to SP1 (much
        # slower). argmin is the least-bad on both axes.
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

    def _safe_estimate(self, parallel_spec: ParallelSpec, seq_len: int, stage: str) -> float:
        try:
            return float(self.task_runtime_estimator(parallel_spec, int(seq_len), stage))
        except Exception as exc:  # noqa: BLE001 - estimator must never crash scheduling
            # Return 0.0 (deadline treated as met) but warn once so a broken
            # estimator does not silently collapse the request to the smallest SP.
            warn_key = (getattr(parallel_spec, "sp", None), stage)
            if warn_key not in self._estimate_warned:
                self._estimate_warned.add(warn_key)
                logger.warning(
                    "runtime_v2 edf_best_fit cost estimate failed for sp=%s stage=%s (%s); "
                    "treating as 0.0 — scheduling will degrade toward the smallest SP.",
                    warn_key[0], stage, exc,
                )
            return 0.0
