# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for runtime_v2 scheduler policies."""

from __future__ import annotations

from collections.abc import Callable

from vllm_omni.diffusion.runtime_v2.protocol import (
    InferenceTask,
    ParallelSpec,
    RequestExecutionPlan,
)
from vllm_omni.diffusion.runtime_v2.topology import RuntimeTopology

TaskRuntimeEstimator = Callable[[ParallelSpec, int, str], float]


_SRTF_STAGE_BASE_COST: dict[str, float] = {
    "text_encode": 1.0,
    "dit_prepare": 0.2,
    "timestep_prepare": 0.05,
    "dit_step_chunk": 1.0,
    "dit_step_chunk_cfg": 1.0,
    "dit_step_chunk_nocfg": 0.5,
    "vae_decode": 0.3,
    "finalize": 0.01,
}

_SRTF_STAGE_WORKLOAD_EXPONENT: dict[str, float] = {
    # Text encoder is closer to linear/super-linear with sequence length.
    "text_encode": 1.2,
    # DiT-family stages are attention-heavy; use quadratic default.
    "dit_prepare": 1.0,
    "dit_step_chunk": 2.0,
    "dit_step_chunk_cfg": 2.0,
    "dit_step_chunk_nocfg": 2.0,
    # Other glue/control stages.
    "timestep_prepare": 1.0,
    "vae_decode": 1.3,
    "finalize": 1.0,
}


def default_srtf_task_runtime_estimator(
    parallel_spec: ParallelSpec,
    seq_len: int,
    request_stage: str,
) -> float:
    """Stage-aware default estimator for runtime_v2 SRTF.

    The estimate is intentionally coarse but stable:
    - stage-specific base weights encode relative kernel cost
    - stage-specific exponents model non-linear scaling (DiT defaults to quadratic)
    - effective parallelism gets sublinear speedup
    """
    stage = str(request_stage).lower()
    base_cost = _SRTF_STAGE_BASE_COST.get(stage, 1.0)
    workload_exponent = _SRTF_STAGE_WORKLOAD_EXPONENT.get(stage, 1.0)
    workload = float(max(1, seq_len))
    effective_parallel = max(1, parallel_spec.tp) * max(1, parallel_spec.sp)
    if parallel_spec.cfg_parallel:
        effective_parallel *= max(1, parallel_spec.cfg)
    # Sublinear speedup is usually a better default for scheduling than ideal linear scaling.
    parallel_speedup = float(effective_parallel) ** 0.8
    return max(1e-6, base_cost * (workload**workload_exponent) / parallel_speedup)


def _resolve_seq_len(task: InferenceTask, plan: RequestExecutionPlan) -> int:
    for key in ("seqlen", "seq_len", "sequence_length", "max_sequence_length"):
        value = task.payload.get(key)
        if isinstance(value, int) and value > 0:
            return value
    for key in ("seqlen", "seq_len", "sequence_length", "max_sequence_length"):
        value = plan.metadata.get(key)
        if isinstance(value, int) and value > 0:
            return value
    if task.step_range is not None:
        return task.step_range.end - task.step_range.start
    return 1


def _resolve_request_stage(task: InferenceTask) -> str:
    for key in ("cost_model_stage", "request_stage", "req_stage", "stage"):
        value = task.payload.get(key)
        if isinstance(value, str) and value:
            return value
    return task.kind.value


def _resolve_explicit_request_group(plan: RequestExecutionPlan) -> str | None:
    explicit_groups = {task.group_id for task in plan.tasks.values() if task.group_id is not None}
    if not explicit_groups:
        return None
    if len(explicit_groups) != 1:
        raise ValueError(
            f"request {plan.request_id} contains tasks pinned to multiple groups: {sorted(explicit_groups)!r}"
        )
    return next(iter(explicit_groups))


def _candidate_groups_for_plan(topology: RuntimeTopology, plan: RequestExecutionPlan) -> tuple[str, ...]:
    return tuple(
        group.group_id
        for group in topology.groups
        if all(task.kind in group.supported_task_kinds for task in plan.tasks.values())
    )
