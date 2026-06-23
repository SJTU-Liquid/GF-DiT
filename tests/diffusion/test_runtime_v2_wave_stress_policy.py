# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import time

import pytest

from vllm_omni.diffusion.runtime_v2.policies.wave_stress import (
    _PINGPONG_CYCLE,
    _WAVE_ROLES,
    WaveStressSchedulerPolicy,
)
from vllm_omni.diffusion.runtime_v2.protocol import (
    ExecutionGroupSpec,
    InferenceTask,
    ParallelSpec,
    RequestExecutionPlan,
    StepRange,
    TaskKind,
    TaskStatus,
    WorkerEvent,
    WorkerEventKind,
)
from vllm_omni.diffusion.runtime_v2.topology import RuntimeTopology, WorkerSpec

pytestmark = [pytest.mark.diffusion, pytest.mark.core_model, pytest.mark.cpu]


def _topology() -> RuntimeTopology:
    return RuntimeTopology(
        workers=tuple(WorkerSpec(worker_rank=r, device_id=r) for r in range(4)),
        groups=(
            ExecutionGroupSpec(
                group_id="g_base",
                ranks=(0, 1, 2, 3),
                parallel_spec=ParallelSpec(tp=1, sp=4, cfg=1),
                supported_task_kinds=(
                    TaskKind.TEXT_ENCODE,
                    TaskKind.DIT_PREPARE,
                    TaskKind.TIMESTEP_PREPARE,
                    TaskKind.DIT_STEP_CHUNK,
                    TaskKind.VAE_DECODE,
                    TaskKind.FINALIZE,
                ),
                ulysses_degree=4,
                ring_degree=1,
            ),
        ),
    )


def _make_dit_only_plan(request_id: str, num_chunks: int = 1) -> RequestExecutionPlan:
    """A request whose only tasks are DIT_STEP_CHUNK so dispatch ordering is
    decided solely by the wave-stress placement, not by stages we don't care
    about."""
    tasks: dict[str, InferenceTask] = {}
    prev_id: str | None = None
    for i in range(num_chunks):
        tid = f"{request_id}:chunk{i}"
        deps = (prev_id,) if prev_id else ()
        tasks[tid] = InferenceTask(
            task_id=tid,
            request_id=request_id,
            kind=TaskKind.DIT_STEP_CHUNK,
            group_id=None,
            parallel_spec=ParallelSpec(),
            dependencies=deps,
            step_range=StepRange(start=i, end=i + 1),
        )
        prev_id = tid
    terminal = (prev_id,) if prev_id else ()
    return RequestExecutionPlan(
        request_id=request_id,
        tasks=tasks,
        terminal_task_ids=terminal,
    )


def _exec_end(task: InferenceTask) -> WorkerEvent:
    return WorkerEvent(
        event_id=f"{task.task_id}:exec_end",
        task_id=task.task_id,
        request_id=task.request_id,
        group_id=task.group_id or "",
        worker_rank=0,
        kind=WorkerEventKind.TASK_EXEC_END,
        timestamp_ns=time.monotonic_ns(),
    )


def _request_finished(request_id: str) -> WorkerEvent:
    return WorkerEvent(
        event_id=f"{request_id}:finished",
        task_id="",
        request_id=request_id,
        group_id="",
        worker_rank=0,
        kind=WorkerEventKind.REQUEST_FINISHED,
        timestamp_ns=time.monotonic_ns(),
    )


def test_wave_stress_rejects_explicit_non_gfc_backend() -> None:
    with pytest.raises(RuntimeError, match="runtime_v2_collective_backend='gfc'"):
        WaveStressSchedulerPolicy(_topology(), collective_backend="torch")


def test_wave_stress_accepts_explicit_gfc_backend() -> None:
    WaveStressSchedulerPolicy(_topology(), collective_backend="gfc")


def test_wave_stress_registers_subgroups_into_topology() -> None:
    topology = _topology()
    WaveStressSchedulerPolicy(topology, require_gfc_backend=False)
    g01 = topology.get_group("stress_g01")
    g12 = topology.get_group("stress_g12")
    gfull = topology.get_group("stress_gfull")
    assert g01.ranks == (0, 1) and int(g01.parallel_spec.sp) == 2
    assert g12.ranks == (1, 2) and int(g12.parallel_spec.sp) == 2
    assert gfull.ranks == (0, 1, 2, 3) and int(gfull.parallel_spec.sp) == 4


def test_wave_stress_holds_wave_requests_until_full_then_assigns_roles() -> None:
    topology = _topology()
    policy = WaveStressSchedulerPolicy(
        topology, warmup_reqs=0, wave_size=4, require_gfc_backend=False
    )

    plans = [_make_dit_only_plan(f"req{i}", num_chunks=2) for i in range(4)]

    # First three submissions are held — nothing dispatched yet.
    for plan in plans[:3]:
        assert list(policy.on_request_submitted(plan)) == []
    assert not policy._active_wave_order
    assert len(policy._held_plans) == 3

    # Fourth submission completes the wave, triggers role assignment, and
    # returns the first round-robin DIT chunk.
    out = list(policy.on_request_submitted(plans[3]))
    assert policy._active_wave_order == [f"req{i}" for i in range(4)]
    for i, role in enumerate(_WAVE_ROLES):
        assert policy._req_role[f"req{i}"] == role

    # Wave dispatch is one-task-at-a-time (rank-locking serializes). The first
    # task out should belong to req0 (cross_g01) which holds ranks (0, 1).
    assert len(out) == 1
    first = out[0]
    assert first.request_id == "req0"
    assert first.group_id == "stress_g01"


def test_wave_stress_round_robin_cycles_through_wave_requests() -> None:
    """End-to-end: drive two laps of round-robin dispatch across the four
    wave requests by manually playing the role the real scheduler plays
    (release follow-up chunks via on_tasks_runnable when their predecessor
    finishes)."""
    topology = _topology()
    policy = WaveStressSchedulerPolicy(
        topology, warmup_reqs=0, wave_size=4, require_gfc_backend=False
    )
    plans = [_make_dit_only_plan(f"req{i}", num_chunks=2) for i in range(4)]
    # Build a quick chunk0 -> chunk1 successor map per request so the test
    # can release chunk1 after chunk0 finishes (mirroring the scheduler's
    # dependency walk, which the policy itself does not do).
    successor_by_task: dict[str, InferenceTask] = {}
    for plan in plans:
        chunks = list(plan.tasks.values())
        for prev, nxt in zip(chunks, chunks[1:], strict=False):
            successor_by_task[prev.task_id] = nxt

    pending: list[InferenceTask] = []
    for plan in plans:
        pending.extend(policy.on_request_submitted(plan))

    dispatch_order: list[tuple[str, str]] = []
    while pending:
        task = pending.pop(0)
        dispatch_order.append((task.request_id, task.group_id or ""))
        nxt = successor_by_task.get(task.task_id)
        if nxt is not None:
            pending.extend(policy.on_tasks_runnable([nxt]))
        pending.extend(policy.on_worker_event(_exec_end(task)))

    assert len(dispatch_order) == 8
    expected_group_for = {
        "req0": "stress_g01",
        "req1": "stress_g12",
        "req2": "stress_gfull",
    }
    first_lap = dispatch_order[:4]
    assert [r for r, _ in first_lap] == ["req0", "req1", "req2", "req3"]
    for rid, gid in first_lap[:3]:
        assert gid == expected_group_for[rid]
    assert first_lap[3] == ("req3", _PINGPONG_CYCLE[0])

    second_lap = dispatch_order[4:]
    assert [r for r, _ in second_lap] == ["req0", "req1", "req2", "req3"]
    for rid, gid in second_lap[:3]:
        assert gid == expected_group_for[rid]
    assert second_lap[3] == ("req3", _PINGPONG_CYCLE[1])


def test_wave_stress_warmup_requests_bypass_wave() -> None:
    topology = _topology()
    policy = WaveStressSchedulerPolicy(
        topology, warmup_reqs=2, wave_size=4, require_gfc_backend=False
    )

    # Warmup plan tasks need an explicit group_id under DynamicStepFCFS — the
    # parent policy errors on group_id=None for non-RESHARD tasks unless a
    # dit_step_schedule is set. Mirror that by pinning warmup plans to g_base.
    warmup_plan = _make_dit_only_plan("warmup0", num_chunks=1)
    for task in warmup_plan.tasks.values():
        task.group_id = "g_base"

    out = list(policy.on_request_submitted(warmup_plan))
    assert len(out) == 1
    assert out[0].request_id == "warmup0"
    assert out[0].group_id == "g_base"
    assert "warmup0" not in policy._req_role


def test_wave_stress_flushes_trailing_partial_pool_after_drain() -> None:
    """When the active wave drains and the collect pool has 1..wave_size-1
    held plans, those trailing plans must be released through the warmup
    path instead of hanging forever waiting for siblings that may never
    arrive."""
    topology = _topology()
    policy = WaveStressSchedulerPolicy(
        topology, warmup_reqs=0, wave_size=4, require_gfc_backend=False
    )

    # 6 prompts: 4 form the active wave, 2 trail behind.
    plans = [_make_dit_only_plan(f"req{i}", num_chunks=1) for i in range(6)]

    pending: list[InferenceTask] = []
    for plan in plans:
        pending.extend(policy.on_request_submitted(plan))
    assert policy._active_wave_order == [f"req{i}" for i in range(4)]
    assert len(policy._held_plans) == 2

    finished_reqs: set[str] = set()
    dispatch_order: list[str] = []
    while pending:
        task = pending.pop(0)
        dispatch_order.append(task.request_id)
        pending.extend(policy.on_worker_event(_exec_end(task)))
        if task.request_id not in finished_reqs:
            finished_reqs.add(task.request_id)
            pending.extend(policy.on_worker_event(_request_finished(task.request_id)))

    # All 6 requests dispatched: 4 as a wave, 2 flushed as warmup.
    assert set(dispatch_order) == {f"req{i}" for i in range(6)}
    assert policy._held_plans == []
    assert policy._active_wave_order == []
    # Trailing plans go through the warmup branch -> _aux_group_id (stress_gfull).
    assert finished_reqs == {f"req{i}" for i in range(6)}


def test_wave_stress_activates_next_wave_after_drain() -> None:
    topology = _topology()
    policy = WaveStressSchedulerPolicy(
        topology, warmup_reqs=0, wave_size=2, require_gfc_backend=False
    )

    plans = [_make_dit_only_plan(f"req{i}", num_chunks=1) for i in range(4)]

    pending: list[InferenceTask] = []
    pending.extend(policy.on_request_submitted(plans[0]))
    pending.extend(policy.on_request_submitted(plans[1]))
    # While first wave is active, submit the next two — they should be held.
    pending.extend(policy.on_request_submitted(plans[2]))
    pending.extend(policy.on_request_submitted(plans[3]))
    assert policy._active_wave_order == ["req0", "req1"]
    assert len(policy._held_plans) == 2

    # Drain by finishing every dispatched task. The second wave should
    # activate automatically once the first wave's requests are done, and
    # its tasks should also flow through.
    finished_reqs: set[str] = set()
    dispatch_order: list[str] = []
    while pending:
        task = pending.pop(0)
        dispatch_order.append(task.request_id)
        pending.extend(policy.on_worker_event(_exec_end(task)))
        if task.request_id not in finished_reqs:
            finished_reqs.add(task.request_id)
            pending.extend(policy.on_worker_event(_request_finished(task.request_id)))

    # All four requests dispatched (first wave + auto-activated second wave).
    assert dispatch_order == ["req0", "req1", "req2", "req3"]
    # Roles were re-assigned for the second wave starting from cross_g01.
    assert finished_reqs == {"req0", "req1", "req2", "req3"}
    assert policy._held_plans == []
    assert policy._active_wave_order == []
